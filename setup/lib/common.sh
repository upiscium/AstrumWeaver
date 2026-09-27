#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

log() {
  printf '[astrumweaver-setup] %s\n' "$*" >&2
}

die() {
  printf '[astrumweaver-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_root() {
  local root="${1:-/}"
  [[ -n "$root" ]] || die "root must not be empty"
  if [[ "$root" != "/" ]]; then
    mkdir -p "$root"
    root="$(cd "$root" && pwd)"
  fi
  printf '%s\n' "$root"
}

root_path() {
  local root="$1" path="$2"
  if [[ "$root" == "/" ]]; then
    printf '%s\n' "$path"
  else
    printf '%s%s\n' "$root" "$path"
  fi
}

install_same_or_fail() {
  local source="$1" destination="$2" mode="$3"
  [[ -f "$source" ]] || die "source file does not exist: $source"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" ]] || die "destination exists but is not a file: $destination"
    if cmp -s "$source" "$destination"; then
      chmod "$mode" "$destination"
      return 0
    fi
    die "destination differs; refusing overwrite: $destination"
  fi
  install -D -m "$mode" "$source" "$destination"
}

install_generated_same_or_fail() {
  local source="$1" destination="$2" mode="$3"
  install_same_or_fail "$source" "$destination" "$mode"
}

validate_role_unit_identity() {
  local unit="$1" expected_user="$2"

  # A role that has not been installed yet has no identity to validate.  A
  # dangling unit link, on the other hand, is an installed-but-unreadable
  # state and must not be treated as a fresh installation.
  if [[ ! -e "$unit" ]]; then
    [[ -L "$unit" ]] && die "cannot inspect existing role unit (dangling link): $unit"
    return 0
  fi
  [[ -f "$unit" ]] || die "existing role unit is not a regular file: $unit"

  local -a users=() groups=()
  mapfile -t users < <(
    sed -n -E \
      -e 's/^[[:space:]]*User[[:space:]]*=[[:space:]]*(.*)$/\1/p' \
      "$unit" | sed -E 's/[[:space:]]+$//'
  )
  mapfile -t groups < <(
    sed -n -E \
      -e 's/^[[:space:]]*Group[[:space:]]*=[[:space:]]*(.*)$/\1/p' \
      "$unit" | sed -E 's/[[:space:]]+$//'
  )

  if ((${#users[@]} != 1)) || ((${#groups[@]} != 1)); then
    die "cannot safely determine service identity in existing role unit $unit: exactly one non-empty User= and Group= is required"
  fi

  local unit_user="${users[0]}" unit_group="${groups[0]}"
  if [[ -z "$unit_user" || -z "$unit_group" ]]; then
    die "cannot safely determine service identity in existing role unit $unit: User= and Group= must be non-empty"
  fi

  if [[ "$unit_user" != "$expected_user" || "$unit_group" != "$expected_user" ]]; then
    die "conflicting existing role unit identity in $unit: found User=$unit_user Group=$unit_group, requested User=$expected_user Group=$expected_user"
  fi
}

validate_shared_service_identity() {
  local root="$1" expected_user="$2"
  local unit_dir
  unit_dir="$(root_path "$root" /etc/systemd/system)"

  # Both files are checked on every invocation, including the role being
  # reinstalled.  This prevents a --user change from silently migrating an
  # installed role and makes the two shared service paths single-identity.
  validate_role_unit_identity \
    "$unit_dir/astrumweaver-control.service" "$expected_user"
  validate_role_unit_identity \
    "$unit_dir/astrumweaver-worker.service" "$expected_user"

  # Operator identity overrides are outside the shared-account contract.
  # Do not chown shared paths using only the base unit while a drop-in can
  # make the effective service run under another identity.
  local role dropin
  for role in control worker; do
    for dropin in "$unit_dir/astrumweaver-$role.service.d/"*.conf; do
      [[ -e "$dropin" || -L "$dropin" ]] || continue
      [[ -f "$dropin" ]] || die "cannot inspect service drop-in: $dropin"
      if grep -Eq '^[[:space:]]*(User|Group|DynamicUser)[[:space:]]*=' "$dropin"; then
        die "service identity drop-ins require manual reconciliation before setup: $dropin"
      fi
    done
  done
}

validate_existing_live_service_user() {
  local user="$1"
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"

  if ! id "$user" >/dev/null 2>&1; then
    # A not-yet-created account is handled by ensure_live_service_user after
    # all role-unit identity checks have passed.
    return 0
  fi

  require_cmd getent
  getent group "$user" >/dev/null 2>&1 \
    || die "existing service account '$user' has no same-name service group '$user'; refusing setup"

  local groups group found=0
  groups="$(id -Gn "$user" 2>/dev/null)" \
    || die "cannot determine group membership for existing service account '$user'; refusing setup"
  while IFS= read -r group; do
    if [[ "$group" == "$user" ]]; then
      found=1
    fi
  done < <(printf '%s\n' "$groups" | tr '[:space:]' '\n')
  if [[ "$found" != 1 ]]; then
    die "existing service account '$user' is not a member of same-name service group '$user'; refusing setup"
  fi
}

ensure_live_service_user() {
  local user="$1"
  validate_existing_live_service_user "$user"
  if id "$user" >/dev/null 2>&1; then
    return 0
  fi
  require_cmd useradd
  useradd --system --user-group \
    --home-dir /var/lib/astrumweaver --create-home \
    --shell /usr/sbin/nologin "$user"
  validate_existing_live_service_user "$user"
}

reconcile_service_directories() {
  local root="$1" user="$2" etc_dir="$3" state_dir="$4"
  [[ -d "$etc_dir" ]] || die "service configuration directory is missing: $etc_dir"
  [[ -d "$state_dir" ]] || die "service state directory is missing: $state_dir"

  # The configuration directory is root-owned and group-readable by both
  # roles.  State remains service-owned so the daemons can write it; both
  # roles use the same account/group contract before reaching this point.
  if [[ "$root" == "/" ]]; then
    chown "root:$user" "$etc_dir"
    chown "$user:$user" "$state_dir"
  fi
  chmod 0750 "$etc_dir" "$state_dir"
}

reconcile_service_file() {
  local root="$1" user="$2" path="$3"
  if [[ ! -e "$path" ]]; then
    [[ -L "$path" ]] && die "service file is a dangling link: $path"
    return 0
  fi
  [[ -f "$path" ]] || die "service file is not a regular file: $path"

  if [[ "$root" == "/" ]]; then
    chown "root:$user" "$path"
  fi
  chmod 0640 "$path"
}

resolve_executable() {
  local requested="$1" root="$2" default_command="$3"
  if [[ -n "$requested" ]]; then
    [[ "$requested" == /* ]] || die "--executable must be an absolute path"
    if [[ "$root" == "/" ]]; then
      [[ -x "$requested" ]] || die "executable is not executable: $requested"
    fi
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ "$root" != "/" ]]; then
    die "--executable is required for staged --root installs"
  fi

  local resolved
  resolved="$(command -v "$default_command" || true)"
  [[ -n "$resolved" && -x "$resolved" ]] \
    || die "packaged executable not found on PATH: $default_command"
  printf '%s\n' "$resolved"
}

render_unit() {
  local template="$1" destination="$2" executable="$3" user="$4" runtime_arg="${5:-}"
  local temporary
  temporary="$(mktemp)"
  sed \
    -e "s|@EXECUTABLE@|$executable|g" \
    -e "s|@USER@|$user|g" \
    -e "s|@RUNTIME_ARG@|$runtime_arg|g" \
    "$template" >"$temporary"
  install_same_or_fail "$temporary" "$destination" 0644
  rm -f "$temporary"
}

systemd_reload_and_maybe_start() {
  local service="$1" start="$2"
  require_cmd systemctl
  systemctl daemon-reload
  if [[ "$start" == 1 ]]; then
    systemctl enable --now "$service"
  fi
}
