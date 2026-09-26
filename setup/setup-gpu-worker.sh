#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/lib/common.sh"

CONFIG_SOURCE=''
ENV_SOURCE=''
RUNTIME_MANIFEST_SOURCE=''
EXECUTABLE=''
ROOT='/'
START=0
SERVICE_USER='astrumweaver'
GPU_ISOLATION='auto'
GPU_UUIDS=()
GPU_DEVICE_ENTRIES=()

usage() {
  cat <<'USAGE'
Usage: setup-gpu-worker.sh --config FILE --executable ABSOLUTE_PATH \
  --gpu-uuid UUID [--gpu-uuid UUID ...] [options]

Options:
  --environment-file FILE   Optional systemd EnvironmentFile source.
  --runtime-manifest FILE    Reviewed RuntimeProvider deployment manifest.
  --executable PATH         Optional absolute daemon override; defaults to astrumweaver-worker on PATH.
  --gpu-uuid UUID           Expected NVIDIA GPU UUID; may be repeated.
  --gpu-isolation MODE      GPU device-cgroup policy: auto|on|off (default: auto).
  --gpu-device UUID=PATH    Reviewed UUID to /dev/nvidiaN mapping; may be repeated.
  --root DIR                Stage files below DIR instead of live /.
  --user NAME               Service account name (default: astrumweaver).
  --start                   Enable and start the service after installation.
  -h, --help                Show this help.

The node and GPU exposure must already exist. The script does not configure
Proxmox, PCI passthrough, IOMMU, or host NVIDIA drivers. When GPU isolation is
enabled it configures only the Worker systemd service device allow-list.
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
    --runtime-manifest)
      (($# >= 2)) || die "--runtime-manifest requires a value"
      RUNTIME_MANIFEST_SOURCE="$2"; shift 2 ;;
    --executable)
      (($# >= 2)) || die "--executable requires a value"
      EXECUTABLE="$2"; shift 2 ;;
    --gpu-uuid)
      (($# >= 2)) || die "--gpu-uuid requires a value"
      GPU_UUIDS+=("$2"); shift 2 ;;
    --gpu-isolation)
      (($# >= 2)) || die "--gpu-isolation requires a value"
      GPU_ISOLATION="$2"; shift 2 ;;
    --gpu-device)
      (($# >= 2)) || die "--gpu-device requires a value"
      GPU_DEVICE_ENTRIES+=("$2"); shift 2 ;;
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
if [[ -n "$RUNTIME_MANIFEST_SOURCE" ]]; then
  [[ -f "$RUNTIME_MANIFEST_SOURCE" ]] || die "runtime manifest does not exist: $RUNTIME_MANIFEST_SOURCE"
fi
[[ "$GPU_ISOLATION" =~ ^(auto|on|off)$ ]] || die "--gpu-isolation must be auto, on, or off"
((${#GPU_UUIDS[@]} > 0)) || die "at least one --gpu-uuid is required"
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user"

for uuid in "${GPU_UUIDS[@]}"; do
  [[ -n "$uuid" ]] || die "GPU UUID must not be blank"
  [[ "$uuid" != *[[:space:]]* ]] || die "GPU UUID must not contain whitespace"
done

ROOT="$(normalize_root "$ROOT")"
EXECUTABLE="$(resolve_executable "$EXECUTABLE" "$ROOT" astrumweaver-worker)"

ETC_DIR="$(root_path "$ROOT" /etc/astrumweaver)"
STATE_DIR="$(root_path "$ROOT" /var/lib/astrumweaver)"
UNIT_DIR="$(root_path "$ROOT" /etc/systemd/system)"
DROPIN_DIR="$UNIT_DIR/astrumweaver-worker.service.d"
LIBEXEC_DIR="$(root_path "$ROOT" /usr/local/libexec/astrumweaver)"
CONFIG_DEST="$ETC_DIR/worker.toml"
ENV_DEST="$ETC_DIR/worker.env"
RUNTIME_MANIFEST_DEST="$ETC_DIR/runtime-deployment.json"
GPU_UUID_DEST="$ETC_DIR/gpu-uuids"
GPU_DEVICE_MAP_DEST="$ETC_DIR/gpu-device-map"
PREFLIGHT_DEST="$LIBEXEC_DIR/gpu-preflight"
DEVICE_MAP_HELPER_DEST="$LIBEXEC_DIR/gpu-device-map"
UNIT_DEST="$UNIT_DIR/astrumweaver-worker.service"
ISOLATION_UNIT_DEST="$UNIT_DIR/astrumweaver-worker-gpu-isolation-preflight.service"
ISOLATION_DROPIN_DEST="$DROPIN_DIR/10-gpu-isolation.conf"

install -d -m 0750 "$ETC_DIR" "$STATE_DIR"
install -d -m 0755 "$UNIT_DIR" "$LIBEXEC_DIR"

expected_tmp="$(mktemp)"
device_map_tmp="$(mktemp)"
isolation_unit_tmp="$(mktemp)"
isolation_dropin_tmp="$(mktemp)"
cleanup() {
  rm -f "$expected_tmp" "$device_map_tmp" "$isolation_unit_tmp" "$isolation_dropin_tmp"
}
trap cleanup EXIT

printf '%s\n' "${GPU_UUIDS[@]}" | sort >"$expected_tmp"
if [[ -n "$(uniq -d "$expected_tmp")" ]]; then
  die "duplicate --gpu-uuid values are not allowed"
fi

if ((${#GPU_DEVICE_ENTRIES[@]} > 0)); then
  for entry in "${GPU_DEVICE_ENTRIES[@]}"; do
    [[ "$entry" == *=* ]] || die "--gpu-device must use UUID=/dev/nvidiaN"
    uuid="${entry%%=*}"
    path="${entry#*=}"
    [[ -n "$uuid" && "$path" =~ ^/dev/nvidia[0-9]+$ ]] || die "--gpu-device must use UUID=/dev/nvidiaN"
    printf '%s=%s\n' "$uuid" "$path"
  done | sort >"$device_map_tmp"
  cut -d= -f1 "$device_map_tmp" >"${device_map_tmp}.keys"
  if ! cmp -s "$expected_tmp" "${device_map_tmp}.keys"; then
    rm -f "${device_map_tmp}.keys"
    die "--gpu-device UUID keys must exactly match --gpu-uuid values"
  fi
  rm -f "${device_map_tmp}.keys"
  if [[ -n "$(cut -d= -f2 "$device_map_tmp" | sort | uniq -d)" ]]; then
    die "--gpu-device paths must be unique"
  fi
fi

install_same_or_fail "$CONFIG_SOURCE" "$CONFIG_DEST" 0640
if [[ -n "$ENV_SOURCE" ]]; then
  install_same_or_fail "$ENV_SOURCE" "$ENV_DEST" 0640
fi
if [[ -n "$RUNTIME_MANIFEST_SOURCE" ]]; then
  install_same_or_fail "$RUNTIME_MANIFEST_SOURCE" "$RUNTIME_MANIFEST_DEST" 0640
fi
install_same_or_fail "$expected_tmp" "$GPU_UUID_DEST" 0640
install_same_or_fail "$REPO_ROOT/libexec/gpu-preflight" "$PREFLIGHT_DEST" 0755
install_same_or_fail "$REPO_ROOT/libexec/gpu-device-map" "$DEVICE_MAP_HELPER_DEST" 0755

runtime_arg=''
if [[ -n "$RUNTIME_MANIFEST_SOURCE" ]]; then
  runtime_arg='--runtime-manifest /etc/astrumweaver/runtime-deployment.json'
fi

render_unit \
  "$REPO_ROOT/systemd/astrumweaver-worker.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER" \
  "$runtime_arg"

if [[ "$ROOT" == "/" ]]; then
  ensure_live_service_user "$SERVICE_USER"
  require_cmd nvidia-smi

  exact_set=0
  if "$PREFLIGHT_DEST" "$GPU_UUID_DEST" >/dev/null 2>&1; then
    exact_set=1
  fi

  isolation_enabled=0
  case "$GPU_ISOLATION" in
    on) isolation_enabled=1 ;;
    off) isolation_enabled=0 ;;
    auto)
      if [[ "$exact_set" == 1 ]]; then
        isolation_enabled=0
      else
        isolation_enabled=1
      fi
      ;;
  esac

  if [[ "$isolation_enabled" == 1 ]]; then
    if [[ ! -s "$device_map_tmp" ]]; then
      "$DEVICE_MAP_HELPER_DEST" discover "$GPU_UUID_DEST" >"$device_map_tmp"
    fi
    install_same_or_fail "$device_map_tmp" "$GPU_DEVICE_MAP_DEST" 0640
    "$DEVICE_MAP_HELPER_DEST" verify "$GPU_UUID_DEST" "$GPU_DEVICE_MAP_DEST"
  else
    "$PREFLIGHT_DEST" "$GPU_UUID_DEST"
    if [[ -e "$ISOLATION_DROPIN_DEST" || -e "$ISOLATION_UNIT_DEST" || -e "$GPU_DEVICE_MAP_DEST" ]]; then
      die "existing GPU isolation state requires reviewed removal before --gpu-isolation off"
    fi
  fi

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
      die "service account cannot access NVIDIA devices"
    fi
  fi
else
  isolation_enabled=0
  if [[ "$GPU_ISOLATION" == "on" || -s "$device_map_tmp" ]]; then
    [[ -s "$device_map_tmp" ]] || die "staged GPU isolation requires --gpu-device UUID=/dev/nvidiaN entries"
    isolation_enabled=1
    install_same_or_fail "$device_map_tmp" "$GPU_DEVICE_MAP_DEST" 0640
  fi

  if [[ "$START" == 1 ]]; then
    die "--start cannot be used with staged --root installs"
  fi
fi

if [[ "$isolation_enabled" == 1 ]]; then
  install -d -m 0755 "$DROPIN_DIR"

  cat >"$isolation_unit_tmp" <<'EOF'
[Unit]
Description=Verify AstrumWeaver Worker GPU UUID/device mapping
Before=astrumweaver-worker.service

[Service]
Type=oneshot
ExecStart=/usr/local/libexec/astrumweaver/gpu-device-map verify /etc/astrumweaver/gpu-uuids /etc/astrumweaver/gpu-device-map
EOF
  install_generated_same_or_fail "$isolation_unit_tmp" "$ISOLATION_UNIT_DEST" 0644

  {
    printf '[Unit]\n'
    printf 'Requires=astrumweaver-worker-gpu-isolation-preflight.service\n'
    printf 'After=astrumweaver-worker-gpu-isolation-preflight.service\n\n'
    printf '[Service]\n'
    printf 'DevicePolicy=closed\n'
    while IFS='=' read -r uuid path; do
      printf 'DeviceAllow=%s rw\n' "$path"
    done <"$GPU_DEVICE_MAP_DEST"

    for path in \
      /dev/nvidiactl \
      /dev/nvidia-modeset \
      /dev/nvidia-uvm \
      /dev/nvidia-uvm-tools \
      /dev/nvidia-nvswitchctl; do
      if [[ "$ROOT" != "/" || -e "$path" ]]; then
        printf 'DeviceAllow=%s rw\n' "$path"
      fi
    done

    visible="$(IFS=,; printf '%s' "${GPU_UUIDS[*]}")"
    printf 'Environment=CUDA_VISIBLE_DEVICES=%s\n' "$visible"
  } >"$isolation_dropin_tmp"
  install_generated_same_or_fail "$isolation_dropin_tmp" "$ISOLATION_DROPIN_DEST" 0644
fi

if [[ "$ROOT" == "/" ]]; then
  chown root:"$SERVICE_USER" "$CONFIG_DEST" "$GPU_UUID_DEST"
  if [[ -f "$GPU_DEVICE_MAP_DEST" ]]; then
    chown root:"$SERVICE_USER" "$GPU_DEVICE_MAP_DEST"
  fi
  if [[ -f "$RUNTIME_MANIFEST_DEST" ]]; then
    chown root:"$SERVICE_USER" "$RUNTIME_MANIFEST_DEST"
  fi
  if [[ -f "$ENV_DEST" ]]; then
    chown root:"$SERVICE_USER" "$ENV_DEST"
  fi
  chown "$SERVICE_USER":"$SERVICE_USER" "$STATE_DIR"

  systemd_reload_and_maybe_start astrumweaver-worker.service "$START"
fi
log "GPU worker host integration complete"
