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
Usage: setup-gpu-worker.sh --config FILE --executable ABSOLUTE_PATH [options]

Options:
  --environment-file FILE   Optional systemd EnvironmentFile source.
  --runtime-manifest FILE    Reviewed RuntimeProvider deployment manifest.
  --executable PATH         Optional absolute daemon override; defaults to astrumweaver-worker on PATH.
  GPU ownership is read only from worker.gpu_uuids in --config.
  --gpu-isolation MODE      GPU device-cgroup policy: auto|on|off (default: auto).
  --gpu-device UUID=PATH    Reviewed UUID to /dev/nvidiaN mapping; may be repeated.
  --root DIR                Stage files below DIR instead of live /.
  --user NAME               Worker service account and group (default: astrumweaver).
  --start                   Enable and start the service after installation.
  -h, --help                Show this help.

Control and Worker share the traverse-only /etc/astrumweaver directory, but use
separate role accounts and state directories. Worker remains astrumweaver with
/var/lib/astrumweaver for SystemdSetupDriver compatibility; when both roles are
installed, their --user values must differ. Existing role identity/state
changes are rejected pending an explicit migration review.

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
if [[ "$GPU_ISOLATION" == "off" && ${#GPU_DEVICE_ENTRIES[@]} -gt 0 ]]; then
  die "--gpu-device cannot be used when --gpu-isolation off"
fi
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user"
validate_role_user_name "$SERVICE_USER"

ROOT="$(normalize_root "$ROOT")"
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
GPU_ISOLATION_PROBE_DEST="$LIBEXEC_DIR/gpu-isolation-probe"
GPU_MAPPING_PYTHON_DEST="$LIBEXEC_DIR/gpu_mapping.py"
UNIT_DEST="$UNIT_DIR/astrumweaver-worker.service"
ISOLATION_UNIT_DEST="$UNIT_DIR/astrumweaver-worker-gpu-isolation-preflight.service"
ISOLATION_DROPIN_DEST="$DROPIN_DIR/10-gpu-isolation.conf"
LEGACY_GPU_DEVICE_MAP_SHA256='faa3836dbf190616d0c4ad32deaad1d3dda66ff1d903468095ab33edc6ea735a'
LEGACY_GPU_PREFLIGHT_SHA256='6a807fa50fc944193f98a32778ef37df98db8e6744546c99a2694bc492bfc236'
runtime_arg=''
if [[ -n "$RUNTIME_MANIFEST_SOURCE" ]]; then
  runtime_arg=' --runtime-manifest /etc/astrumweaver/runtime-deployment.json'
fi

validate_role_unit_pair "$ROOT" worker "$SERVICE_USER" astrumweaver
validate_legacy_shared_state_path "$ROOT" worker
EXECUTABLE="$(resolve_executable "$EXECUTABLE" "$ROOT" astrumweaver-worker)"
reject_symlink_path \
  "$ETC_DIR" "$STATE_DIR" "$CONFIG_DEST" "$ENV_DEST" \
  "$RUNTIME_MANIFEST_DEST" "$GPU_UUID_DEST" "$GPU_DEVICE_MAP_DEST" \
  "$PREFLIGHT_DEST" "$DEVICE_MAP_HELPER_DEST" "$GPU_ISOLATION_PROBE_DEST" "$GPU_MAPPING_PYTHON_DEST" "$UNIT_DEST" \
  "$ISOLATION_UNIT_DEST" "$ISOLATION_DROPIN_DEST"
validate_unit_template \
  "$REPO_ROOT/systemd/astrumweaver-worker.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER" \
  "$runtime_arg" \
  worker \
  astrumweaver

declared_tmp="$(mktemp)"
expected_tmp="$(mktemp)"
device_map_tmp="$(mktemp)"
isolation_unit_tmp="$(mktemp)"
isolation_dropin_tmp="$(mktemp)"
device_map_exec_tmp="$(mktemp)"
cleanup() {
  rm -f "$declared_tmp" "$expected_tmp" "$device_map_tmp" "$isolation_unit_tmp" "$isolation_dropin_tmp" "$device_map_exec_tmp"
}
trap cleanup EXIT

GPU_UUID_READER="$REPO_ROOT/libexec/worker-gpu-uuids"
[[ -f "$GPU_UUID_READER" ]] || die "Worker GPU config reader is unavailable: $GPU_UUID_READER"
PREFLIGHT_SOURCE="$REPO_ROOT/libexec/gpu-preflight"
DEVICE_MAP_HELPER_SOURCE="$REPO_ROOT/libexec/gpu-device-map"
GPU_ISOLATION_PROBE_SOURCE="$REPO_ROOT/libexec/gpu-isolation-probe"
[[ -f "$PREFLIGHT_SOURCE" ]] || die "GPU preflight helper is unavailable"
[[ -f "$DEVICE_MAP_HELPER_SOURCE" ]] || die "canonical GPU mapper helper is unavailable"
[[ -f "$GPU_ISOLATION_PROBE_SOURCE" ]] || die "GPU isolation capability probe is unavailable"
GPU_MAPPING_PYTHON_SOURCE="$REPO_ROOT/libexec/gpu_mapping.py"
if [[ ! -f "$GPU_MAPPING_PYTHON_SOURCE" ]]; then
  GPU_MAPPING_PYTHON_SOURCE="$REPO_ROOT/src/astrumweaver/gpu_mapping.py"
fi
[[ -f "$GPU_MAPPING_PYTHON_SOURCE" ]] || die "canonical GPU mapper is unavailable"

cat >"$device_map_exec_tmp" <<EOF
#!/usr/bin/env bash
exec bash "$DEVICE_MAP_HELPER_SOURCE" "\$@"
EOF
chmod 0755 "$device_map_exec_tmp"

normalize_reviewed_legacy_bash_script() {
  local source="$1" normalized="$2" first_line
  [[ -f "$source" && ! -L "$source" ]] || return 1
  IFS= read -r first_line <"$source" || return 1

  if [[ "$first_line" == '#!/usr/bin/env bash' ]]; then
    cp -- "$source" "$normalized"
    return 0
  fi

  if [[ "$first_line" =~ ^#!/nix/store/[0-9a-z]{32}-bash-[A-Za-z0-9._+~-]+/bin/bash$ ]]; then
    {
      printf '%s\n' '#!/usr/bin/env bash'
      tail -n +2 -- "$source"
    } >"$normalized"
    return 0
  fi

  return 1
}

install_reviewed_script_upgrade() {
  local source="$1" destination="$2" mode="$3" legacy_sha256="${4:-}"
  local normalized digest
  [[ -f "$source" ]] || die "source file does not exist: $source"
  reject_symlink_path "$destination"

  if [[ ! -e "$destination" ]]; then
    install -D -m "$mode" "$source" "$destination"
    return 0
  fi

  [[ -f "$destination" ]] || die "destination exists but is not a file: $destination"
  if cmp -s "$source" "$destination"; then
    chmod "$mode" "$destination"
    return 0
  fi

  normalized="$(mktemp)"
  if ! normalize_reviewed_legacy_bash_script "$destination" "$normalized"; then
    rm -f "$normalized"
    die "destination differs; refusing overwrite: $destination"
  fi

  if cmp -s "$source" "$normalized"; then
    rm -f "$normalized"
    log "upgrading reviewed Nix-patched helper: $destination"
    install -D -m "$mode" "$source" "$destination"
    return 0
  fi

  if [[ -n "$legacy_sha256" ]]; then
    require_cmd sha256sum
    digest="$(sha256sum "$normalized")"
    digest="${digest%% *}"
    if [[ "$digest" == "$legacy_sha256" ]]; then
      rm -f "$normalized"
      log "upgrading reviewed legacy helper: $destination"
      install -D -m "$mode" "$source" "$destination"
      return 0
    fi
  fi

  rm -f "$normalized"
  die "destination differs; refusing overwrite: $destination"
}

if [[ -x "$GPU_UUID_READER" ]]; then
  GPU_UUID_READER_CMD=("$GPU_UUID_READER")
else
  require_cmd python3
  GPU_UUID_READER_CMD=(python3 "$GPU_UUID_READER")
fi
if ! "${GPU_UUID_READER_CMD[@]}" --config "$CONFIG_SOURCE" >"$declared_tmp"; then
  die "cannot derive GPU ownership from worker.toml"
fi
mapfile -t GPU_UUIDS <"$declared_tmp"
((${#GPU_UUIDS[@]} > 0)) || die "worker.toml declares no GPU UUIDs"
sort "$declared_tmp" >"$expected_tmp"

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
    die "--gpu-device UUID keys must exactly match worker.toml gpu_uuids"
  fi
  rm -f "${device_map_tmp}.keys"
  if [[ -n "$(cut -d= -f2 "$device_map_tmp" | sort | uniq -d)" ]]; then
    die "--gpu-device paths must be unique"
  fi
fi

if [[ "$ROOT" == "/" ]]; then
  validate_live_service_accounts worker "$SERVICE_USER"
  ensure_live_service_user "$SERVICE_USER" "$STATE_DIR" worker
  ensure_service_config_membership "$SERVICE_USER"
  validate_live_service_accounts worker "$SERVICE_USER"
fi

isolation_enabled=0
exact_set=0
if [[ "$ROOT" == "/" ]]; then
  require_cmd nvidia-smi

  if bash "$PREFLIGHT_SOURCE" "$expected_tmp" >/dev/null 2>&1; then
    exact_set=1
  fi

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
      "$device_map_exec_tmp" discover "$expected_tmp" >"$device_map_tmp"
    fi
    "$device_map_exec_tmp" verify "$expected_tmp" "$device_map_tmp"

    if systemctl is-active --quiet astrumweaver-worker.service; then
      die "Worker service is active; stop it before applying GPU isolation"
    else
      service_status=$?
      # systemd returns 3 for an installed inactive unit and 4 when the unit
      # does not exist yet. Both are safe pre-mutation states.
      [[ "$service_status" == 3 || "$service_status" == 4 ]] \
        || die "cannot determine Worker service state"
    fi

    if [[ "$exact_set" == 0 ]]; then
      visible="$(IFS=,; printf '%s' "${GPU_UUIDS[*]}")"
      set +e
      ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND="$device_map_exec_tmp" \
      ASTRUMWEAVER_NVIDIA_SMI="$(command -v nvidia-smi)" \
        bash "$GPU_ISOLATION_PROBE_SOURCE" \
        "$expected_tmp" \
        "$device_map_tmp" \
        "$CONFIG_SOURCE" \
        "$visible"
      isolation_probe_rc=$?
      set -e

      if [[ "$isolation_probe_rc" == 3 ]]; then
        die "GPU subset isolation is not enforceable in this environment; narrow guest-visible GPU exposure externally"
      fi
      [[ "$isolation_probe_rc" == 0 ]] \
        || die "GPU subset isolation capability probe failed"
    fi
  fi
fi

install -d -m 0710 "$ETC_DIR"
install -d -m 0750 "$STATE_DIR"
install -d -m 0755 "$UNIT_DIR" "$LIBEXEC_DIR"
reconcile_service_directories \
  "$ROOT" "$SERVICE_USER" "$CONFIG_GROUP_NAME" "$ETC_DIR" "$STATE_DIR"

install_same_or_fail "$CONFIG_SOURCE" "$CONFIG_DEST" 0640
if [[ -n "$ENV_SOURCE" ]]; then
  install_same_or_fail "$ENV_SOURCE" "$ENV_DEST" 0640
fi
if [[ -n "$RUNTIME_MANIFEST_SOURCE" ]]; then
  install_same_or_fail "$RUNTIME_MANIFEST_SOURCE" "$RUNTIME_MANIFEST_DEST" 0640
fi
install_same_or_fail "$expected_tmp" "$GPU_UUID_DEST" 0640
install_reviewed_script_upgrade "$REPO_ROOT/libexec/gpu-preflight" "$PREFLIGHT_DEST" 0755 "$LEGACY_GPU_PREFLIGHT_SHA256"
install_reviewed_script_upgrade "$REPO_ROOT/libexec/gpu-device-map" "$DEVICE_MAP_HELPER_DEST" 0755 "$LEGACY_GPU_DEVICE_MAP_SHA256"
install_reviewed_script_upgrade "$REPO_ROOT/libexec/gpu-isolation-probe" "$GPU_ISOLATION_PROBE_DEST" 0755
install_same_or_fail "$GPU_MAPPING_PYTHON_SOURCE" "$GPU_MAPPING_PYTHON_DEST" 0640

render_unit \
  "$REPO_ROOT/systemd/astrumweaver-worker.service.in" \
  "$UNIT_DEST" \
  "$EXECUTABLE" \
  "$SERVICE_USER" \
  "$runtime_arg"

if [[ "$ROOT" == "/" ]]; then
  if [[ "$isolation_enabled" == 1 ]]; then
    install_same_or_fail "$device_map_tmp" "$GPU_DEVICE_MAP_DEST" 0640
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
  if [[ "$GPU_ISOLATION" == "off" ]] && {
    [[ -e "$ISOLATION_DROPIN_DEST" || -e "$ISOLATION_UNIT_DEST" || -e "$GPU_DEVICE_MAP_DEST" ]]
  }; then
    die "existing GPU isolation state requires reviewed removal before --gpu-isolation off"
  fi
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
    if [[ "$ROOT" != "/" || "$exact_set" == 0 ]]; then
      printf 'Environment=ASTRUMWEAVER_GPU_PREFLIGHT_MODE=isolated-access\n'
      printf 'Environment=ASTRUMWEAVER_GPU_DEVICE_MAP=/etc/astrumweaver/gpu-device-map\n'
      printf 'Environment=ASTRUMWEAVER_GPU_WORKER_CONFIG=/etc/astrumweaver/worker.toml\n'
      printf 'Environment=ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND=/usr/local/libexec/astrumweaver/gpu-device-map\n'
    fi
  } >"$isolation_dropin_tmp"
  install_generated_same_or_fail "$isolation_dropin_tmp" "$ISOLATION_DROPIN_DEST" 0644
fi

reconcile_service_file "$ROOT" "$SERVICE_USER" "$CONFIG_DEST"
reconcile_service_file "$ROOT" "$SERVICE_USER" "$ENV_DEST"
reconcile_service_file "$ROOT" "$SERVICE_USER" "$RUNTIME_MANIFEST_DEST"
reconcile_service_file "$ROOT" "$SERVICE_USER" "$GPU_UUID_DEST"
reconcile_service_file "$ROOT" "$SERVICE_USER" "$GPU_DEVICE_MAP_DEST"

if [[ "$ROOT" == "/" ]]; then
  systemd_reload_and_maybe_start astrumweaver-worker.service "$START"
fi
log "GPU worker host integration complete"
