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
SERVICE_USER='astrumweaver-control'

usage() {
  cat <<'USAGE'
Usage: setup-control-plane.sh --config FILE [options]

Options:
  --environment-file FILE   Optional systemd EnvironmentFile source.
  --executable PATH         Optional absolute daemon override; defaults to astrumweaver-control on PATH.
  --root DIR                Stage files below DIR instead of live /.
  --user NAME               Control service account and group (default: astrumweaver-control).
  --start                   Enable and start the service after installation.
  -h, --help                Show this help.

Control and Worker share the traverse-only /etc/astrumweaver directory, but use
separate role accounts and state directories. Control defaults to the
astrumweaver-control account and /var/lib/astrumweaver-control; when both roles
are installed, their --user values must differ. Existing role identity/state
changes are rejected pending an explicit migration review.

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
validate_role_user_name "$SERVICE_USER"

ROOT="$(normalize_root "$ROOT")"
ETC_DIR="$(root_path "$ROOT" /etc/astrumweaver)"
STATE_DIR="$(root_path "$ROOT" /var/lib/astrumweaver-control)"
UNIT_DIR="$(root_path "$ROOT" /etc/systemd/system)"
CONFIG_DEST="$ETC_DIR/control.toml"
ENV_DEST="$ETC_DIR/control.env"
UNIT_DEST="$UNIT_DIR/astrumweaver-control.service"

validate_role_unit_pair "$ROOT" control "$SERVICE_USER" astrumweaver-control
validate_legacy_shared_state_path "$ROOT" control
EXECUTABLE="$(resolve_executable "$EXECUTABLE" "$ROOT" astrumweaver-control)"
reject_symlink_path \
  "$ETC_DIR" "$STATE_DIR" "$CONFIG_DEST" "$ENV_DEST" "$UNIT_DEST"
validate_unit_template \
  "$REPO_ROOT/systemd/astrumweaver-control.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER" \
  '' \
  control \
  astrumweaver-control
if [[ "$ROOT" == "/" ]]; then
  validate_live_service_accounts control "$SERVICE_USER"
  ensure_live_service_user "$SERVICE_USER" "$STATE_DIR" control
  ensure_service_config_membership "$SERVICE_USER"
  validate_live_service_accounts control "$SERVICE_USER"
fi

install -d -m 0710 "$ETC_DIR"
install -d -m 0750 "$STATE_DIR"
install -d -m 0755 "$UNIT_DIR"
reconcile_service_directories \
  "$ROOT" "$SERVICE_USER" "$CONFIG_GROUP_NAME" "$ETC_DIR" "$STATE_DIR"
install_same_or_fail "$CONFIG_SOURCE" "$CONFIG_DEST" 0640
if [[ -n "$ENV_SOURCE" ]]; then
  install_same_or_fail "$ENV_SOURCE" "$ENV_DEST" 0640
fi

render_unit \
  "$REPO_ROOT/systemd/astrumweaver-control.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER"

reconcile_service_file "$ROOT" "$SERVICE_USER" "$CONFIG_DEST"
reconcile_service_file "$ROOT" "$SERVICE_USER" "$ENV_DEST"

if [[ "$ROOT" == "/" ]]; then
  systemd_reload_and_maybe_start astrumweaver-control.service "$START"
elif [[ "$START" == 1 ]]; then
  die "--start cannot be used with staged --root installs"
fi

log "control-plane host integration complete"
