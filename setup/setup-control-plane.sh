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

usage() {
  cat <<'USAGE'
Usage: setup-control-plane.sh --config FILE [options]

Options:
  --environment-file FILE   Optional systemd EnvironmentFile source.\n  --executable PATH         Optional absolute daemon override; defaults to astrumweaver-control on PATH.
  --root DIR                Stage files below DIR instead of live /.
  --user NAME               Service account name (default: astrumweaver).
  --start                   Enable and start the service after installation.
  -h, --help                Show this help.

The script configures an existing Linux node only. It never creates a VM/LXC,
network, storage pool, database host, or other infrastructure.
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
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user"

ROOT="$(normalize_root "$ROOT")"
EXECUTABLE="$(resolve_executable "$EXECUTABLE" "$ROOT" astrumweaver-control)"

ETC_DIR="$(root_path "$ROOT" /etc/astrumweaver)"
STATE_DIR="$(root_path "$ROOT" /var/lib/astrumweaver)"
UNIT_DIR="$(root_path "$ROOT" /etc/systemd/system)"
CONFIG_DEST="$ETC_DIR/control.toml"
ENV_DEST="$ETC_DIR/control.env"
UNIT_DEST="$UNIT_DIR/astrumweaver-control.service"

install -d -m 0750 "$ETC_DIR" "$STATE_DIR"
install -d -m 0755 "$UNIT_DIR"
install_same_or_fail "$CONFIG_SOURCE" "$CONFIG_DEST" 0640
if [[ -n "$ENV_SOURCE" ]]; then
  install_same_or_fail "$ENV_SOURCE" "$ENV_DEST" 0640
fi

render_unit \
  "$REPO_ROOT/systemd/astrumweaver-control.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER"

if [[ "$ROOT" == "/" ]]; then
  ensure_live_service_user "$SERVICE_USER"
  chown root:"$SERVICE_USER" "$CONFIG_DEST"
  if [[ -f "$ENV_DEST" ]]; then
    chown root:"$SERVICE_USER" "$ENV_DEST"
  fi
  chown "$SERVICE_USER":"$SERVICE_USER" "$STATE_DIR"
  systemd_reload_and_maybe_start astrumweaver-control.service "$START"
elif [[ "$START" == 1 ]]; then
  die "--start cannot be used with staged --root installs"
fi

log "control-plane host integration complete"
