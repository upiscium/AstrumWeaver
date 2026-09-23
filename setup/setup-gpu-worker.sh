#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/lib/common.sh"

CONFIG_SOURCE=''
ENV_SOURCE=''
EXECUTABLE=''
ROOT='/'
START=0
SERVICE_USER='astrumweaver'
GPU_UUIDS=()

usage() {
  cat <<'USAGE'
Usage: setup-gpu-worker.sh --config FILE --executable ABSOLUTE_PATH \
  --gpu-uuid UUID [--gpu-uuid UUID ...] [options]

Options:
  --environment-file FILE   Optional systemd EnvironmentFile source.
  --gpu-uuid UUID           Expected NVIDIA GPU UUID; may be repeated.
  --root DIR                Stage files below DIR instead of live /.
  --user NAME               Service account name (default: astrumweaver).
  --start                   Enable and start the service after installation.
  -h, --help                Show this help.

The node and GPU exposure must already exist. The script does not configure
Proxmox, PCI passthrough, IOMMU, device cgroups, or host NVIDIA drivers.
USAGE
}

while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || die "--config requires a value"
      CONFIG_SOURCE="$2"; shift 2 ;;
    --environment-file)
      (($# >= 2)) || die "--environment-file requires a value"
      ENV_SOURCE="$2"; shift 2 ;;
    --executable)
      (($# >= 2)) || die "--executable requires a value"
      EXECUTABLE="$2"; shift 2 ;;
    --gpu-uuid)
      (($# >= 2)) || die "--gpu-uuid requires a value"
      GPU_UUIDS+=("$2"); shift 2 ;;
    --root)
      (($# >= 2)) || die "--root requires a value"
      ROOT="$2"; shift 2 ;;
    --user)
      (($# >= 2)) || die "--user requires a value"
      SERVICE_USER="$2"; shift 2 ;;
    --start)
      START=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "unknown option: $1" ;;
  esac
done

[[ -n "$CONFIG_SOURCE" ]] || die "--config is required"
[[ -f "$CONFIG_SOURCE" ]] || die "config file does not exist: $CONFIG_SOURCE"
if [[ -n "$ENV_SOURCE" ]]; then
  [[ -f "$ENV_SOURCE" ]] || die "environment file does not exist: $ENV_SOURCE"
fi
((${#GPU_UUIDS[@]} > 0)) || die "at least one --gpu-uuid is required"
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user"

for uuid in "${GPU_UUIDS[@]}"; do
  [[ -n "$uuid" ]] || die "GPU UUID must not be blank"
  [[ "$uuid" != *[[:space:]]* ]] || die "GPU UUID must not contain whitespace"
done

ROOT="$(normalize_root "$ROOT")"
EXECUTABLE="$(resolve_executable "$EXECUTABLE" "$ROOT")"

ETC_DIR="$(root_path "$ROOT" /etc/astrumweaver)"
STATE_DIR="$(root_path "$ROOT" /var/lib/astrumweaver)"
UNIT_DIR="$(root_path "$ROOT" /etc/systemd/system)"
LIBEXEC_DIR="$(root_path "$ROOT" /usr/local/libexec/astrumweaver)"
CONFIG_DEST="$ETC_DIR/worker.toml"
ENV_DEST="$ETC_DIR/worker.env"
GPU_UUID_DEST="$ETC_DIR/gpu-uuids"
PREFLIGHT_DEST="$LIBEXEC_DIR/gpu-preflight"
UNIT_DEST="$UNIT_DIR/astrumweaver-worker.service"

install -d -m 0750 "$ETC_DIR" "$STATE_DIR"
install -d -m 0755 "$UNIT_DIR" "$LIBEXEC_DIR"

expected_tmp="$(mktemp)"
cleanup() {
  rm -f "$expected_tmp"
}
trap cleanup EXIT

printf '%s\n' "${GPU_UUIDS[@]}" | sort >"$expected_tmp"
if [[ -n "$(uniq -d "$expected_tmp")" ]]; then
  die "duplicate --gpu-uuid values are not allowed"
fi

install_same_or_fail "$CONFIG_SOURCE" "$CONFIG_DEST" 0640
if [[ -n "$ENV_SOURCE" ]]; then
  install_same_or_fail "$ENV_SOURCE" "$ENV_DEST" 0640
fi
install_same_or_fail "$expected_tmp" "$GPU_UUID_DEST" 0640
install_same_or_fail "$REPO_ROOT/libexec/gpu-preflight" "$PREFLIGHT_DEST" 0755

render_unit \
  "$REPO_ROOT/systemd/astrumweaver-worker.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER"

if [[ "$ROOT" == "/" ]]; then
  ensure_live_service_user "$SERVICE_USER"

  require_cmd nvidia-smi
  "$PREFLIGHT_DEST" "$GPU_UUID_DEST"

  if command -v usermod >/dev/null 2>&1; then
    for group in video render; do
      if getent group "$group" >/dev/null 2>&1; then
        usermod --append --groups "$group" "$SERVICE_USER"
      fi
    done
  fi

  if command -v runuser >/dev/null 2>&1; then
    nvidia_smi="$(command -v nvidia-smi)"
    if ! runuser -u "$SERVICE_USER" -- "$nvidia_smi" \
      --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
      die "service account cannot access the configured NVIDIA devices"
    fi
  fi

  chown root:"$SERVICE_USER" "$CONFIG_DEST" "$GPU_UUID_DEST"
  if [[ -f "$ENV_DEST" ]]; then
    chown root:"$SERVICE_USER" "$ENV_DEST"
  fi
  chown "$SERVICE_USER":"$SERVICE_USER" "$STATE_DIR"

  systemd_reload_and_maybe_start astrumweaver-worker.service "$START"
elif [[ "$START" == 1 ]]; then
  die "--start cannot be used with staged --root installs"
fi

log "GPU worker host integration complete"
