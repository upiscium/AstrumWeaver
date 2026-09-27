#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

CONFIG_GROUP_NAME='astrumweaver-config'
CONFIG_GROUP_GID=''
ROLE_PEER_USER=''
ROLE_PEER_GROUP=''
ROLE_PEER_UNIT=''
UNIT_USER=''
UNIT_GROUP=''
UNIT_STATE=''
log() { printf '[astrumweaver-setup] %s\n' "$*" >&2; }
die() { printf '[astrumweaver-setup] ERROR: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

normalize_root() {
  local root="${1:-/}"
  [[ -n "$root" ]] || die "root must not be empty"
  if [[ "$root" != / ]]; then
    mkdir -p "$root"
    root="$(cd "$root" && pwd)"
  fi
  printf '%s\n' "$root"
}
root_path() {
  local root="$1" path="$2"
  if [[ "$root" == / ]]; then printf '%s\n' "$path"; else printf '%s%s\n' "$root" "$path"; fi
}
reject_symlink_path() {
  local path
  for path in "$@"; do
    [[ -L "$path" ]] && die "refusing to follow symlink destination: $path"
  done
  return 0
}
install_same_or_fail() {
  local source="$1" destination="$2" mode="$3"
  [[ -f "$source" ]] || die "source file does not exist: $source"
  reject_symlink_path "$destination"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" ]] || die "destination exists but is not a file: $destination"
    cmp -s "$source" "$destination" || die "destination differs; refusing overwrite: $destination"
    chmod "$mode" "$destination"
    return 0
  fi
  install -D -m "$mode" "$source" "$destination"
}

install_generated_same_or_fail() { install_same_or_fail "$1" "$2" "$3"; }
legacy_service_migration() {
  die "legacy/shared AstrumWeaver service state or identity detected in $1 ($2); stop both services and review migration guidance before rerunning setup"
}
unit_values() {
  local key="$1" unit="$2"
  sed -n -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)$/\\1/p" "$unit" \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}
parse_unit() {
  local unit="$1" context="$2"
  [[ -f "$unit" && ! -L "$unit" ]] || die "existing role unit is not a regular file: $unit"
  local -a users=() groups=() states=() supplementary=() dynamic=()
  mapfile -t users < <(unit_values User "$unit")
  mapfile -t groups < <(unit_values Group "$unit")
  mapfile -t states < <(unit_values StateDirectory "$unit")
  mapfile -t supplementary < <(unit_values SupplementaryGroups "$unit")
  mapfile -t dynamic < <(unit_values DynamicUser "$unit")
  if ((${#users[@]} != 1 || ${#groups[@]} != 1 || ${#states[@]} != 1)); then
    die "cannot safely determine service identity/state in existing role unit $unit: exactly one non-empty User=, Group=, and StateDirectory= is required"
  fi
  if [[ -z "${users[0]}" || -z "${groups[0]}" || -z "${states[0]}" ]]; then
    die "cannot safely determine service identity/state in existing role unit $unit: User=, Group=, and StateDirectory= must be non-empty"
  fi
  UNIT_USER="${users[0]}"
  UNIT_GROUP="${groups[0]}"
  UNIT_STATE="${states[0]}"
  [[ "$UNIT_USER" == "$UNIT_GROUP" ]] \
    || die "invalid service identity in $context: User=$UNIT_USER and Group=$UNIT_GROUP must name the same role-private identity"
  [[ "$UNIT_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] \
    || die "invalid service identity in $context: User/Group '$UNIT_USER' is not a valid role account"
  validate_role_user_name "$UNIT_USER"
  ((${#dynamic[@]} == 0)) || legacy_service_migration "$context" 'DynamicUser= override requires manual reconciliation'
  if ((${#supplementary[@]} != 1)) || [[ "${supplementary[0]:-}" != "$CONFIG_GROUP_NAME" ]]; then
    legacy_service_migration "$context" "base unit must contain exactly SupplementaryGroups=$CONFIG_GROUP_NAME"
  fi
}
validate_unit_dropins() {
  local unit="$1" dropin
  local dir="${unit}.d"
  [[ -L "$dir" ]] && die "cannot inspect service drop-in directory symlink: $dir"
  if [[ -e "$dir" ]]; then [[ -d "$dir" ]] || die "service drop-in path is not a directory: $dir"; fi
  [[ -d "$dir" ]] || return 0
  for dropin in "$dir"/*.conf; do
    [[ -e "$dropin" || -L "$dropin" ]] || continue
    [[ -f "$dropin" && ! -L "$dropin" && -r "$dropin" ]] || die "cannot inspect service drop-in: $dropin"
    if grep -Eq '^[[:space:]]*(User|Group|DynamicUser|StateDirectory|StateDirectoryMode|SupplementaryGroups)[[:space:]]*=' "$dropin"; then
      die "service identity drop-ins (including state/supplementary overrides) require stopping both services and reviewing migration guidance before setup: $dropin"
    fi
  done
}
validate_existing_role_unit() {
  local unit="$1" expected_user="$3" expected_state="$4"
  parse_unit "$unit" "$2"
  if [[ -n "$expected_user" ]]; then
    if [[ "$UNIT_USER" != "$expected_user" || "$UNIT_GROUP" != "$expected_user" ]]; then
      legacy_service_migration "$unit" "found User=$UNIT_USER Group=$UNIT_GROUP, requested User=$expected_user Group=$expected_user"
    fi
  fi
  [[ "$UNIT_STATE" == "$expected_state" ]] \
    || legacy_service_migration "$unit" "found StateDirectory=$UNIT_STATE, expected StateDirectory=$expected_state"
  validate_unit_dropins "$unit"
}
validate_role_unit_pair() {
  local root="$1" role="$2" requested_user="$3" expected_state="$4"
  local dir own peer peer_role peer_state
  dir="$(root_path "$root" /etc/systemd/system)"
  case "$role" in
    control) own="$dir/astrumweaver-control.service"; peer="$dir/astrumweaver-worker.service"; peer_role=worker; peer_state=astrumweaver ;;
    worker) own="$dir/astrumweaver-worker.service"; peer="$dir/astrumweaver-control.service"; peer_role=control; peer_state=astrumweaver-control ;;
    *) die "unknown service role: $role" ;;
  esac
  ROLE_PEER_USER=''; ROLE_PEER_GROUP=''; ROLE_PEER_UNIT=''
  if [[ -e "$own" || -L "$own" ]]; then
    validate_existing_role_unit "$own" "$own" "$requested_user" "$expected_state"
  else
    validate_unit_dropins "$own"
  fi
  if [[ -e "$peer" || -L "$peer" ]]; then
    validate_existing_role_unit "$peer" "$peer" '' "$peer_state"
    ROLE_PEER_USER="$UNIT_USER"; ROLE_PEER_GROUP="$UNIT_GROUP"; ROLE_PEER_UNIT="$peer"
    if [[ "$ROLE_PEER_USER" == "$requested_user" || "$ROLE_PEER_GROUP" == "$requested_user" ]]; then
      legacy_service_migration "$peer" "peer $peer_role uses requested identity User=$ROLE_PEER_USER Group=$ROLE_PEER_GROUP; Control and Worker must use different service users"
    fi
  else
    validate_unit_dropins "$peer"
  fi
}
validate_legacy_shared_state_path() {
  local root="$1" role="$2" legacy
  case "$role" in
    control)
      legacy="$(root_path "$root" /var/lib/astrumweaver)"
      if [[ -e "$legacy" || -L "$legacy" ]] && [[ -z "$ROLE_PEER_UNIT" ]]; then
        legacy_service_migration "$legacy" 'legacy shared state path has no validated Worker unit'
      fi
      ;;
    worker) ;;
    *) die "unknown service role: $role" ;;
  esac
}
render_unit_content() {
  local template="$1" executable="$2" user="$3" runtime_arg="${4:-}"
  sed -e "s|@EXECUTABLE@|$executable|g" -e "s|@USER@|$user|g" -e "s|@RUNTIME_ARG@|$runtime_arg|g" "$template"
}
validate_unit_template() {
  local template="$1" destination="$2" executable="$3" user="$4" runtime_arg="$5" role="$6" expected_state="$7" temporary
  temporary="$(mktemp)"
  render_unit_content "$template" "$executable" "$user" "$runtime_arg" >"$temporary"
  parse_unit "$temporary" "$template"
  [[ "$UNIT_USER" == "$user" && "$UNIT_GROUP" == "$user" && "$UNIT_STATE" == "$expected_state" ]] \
    || die "generated $role service unit does not match requested identity/state: $template"
  reject_symlink_path "$destination"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" ]] || die "destination exists but is not a file: $destination"
    cmp -s "$temporary" "$destination" || { rm -f "$temporary"; die "destination differs; refusing overwrite: $destination"; }
  fi
  rm -f "$temporary"
}
render_unit() {
  local temporary
  temporary="$(mktemp)"
  render_unit_content "$1" "$3" "$4" "${5:-}" >"$temporary"
  install_same_or_fail "$temporary" "$2" 0644
  rm -f "$temporary"
}
validate_role_user_name() {
  case "$1" in
    root|astrumweaver-config|video|render) die "service user '$1' is reserved and cannot own a role-private primary group" ;;
  esac
}
passwd_ids() {
  local record uid gid
  record="$(getent passwd "$1" 2>/dev/null || true)"
  [[ -n "$record" ]] || return 1
  IFS=: read -r _ _ uid gid _ <<<"$record"
  printf '%s:%s\n' "$uid" "$gid"
}
group_gid() {
  local record name gid
  record="$(getent group "$1" 2>/dev/null || true)"
  [[ -n "$record" ]] || return 1
  IFS=: read -r name _ gid _ <<<"$record"
  [[ "$name" == "$1" ]] || return 1
  printf '%s\n' "$gid"
}
group_members() {
  local record members
  record="$(getent group "$1" 2>/dev/null || true)"
  [[ -n "$record" ]] || return 1
  IFS=: read -r _ _ _ members <<<"$record"
  printf '%s\n' "$members"
}
validate_id() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$3 has invalid $1 '$2'; UID/GID must be nonzero"
}
validate_unique_numeric_id() {
  local database="$1" owner="$2" expected="$3" kind records name password number rest found=0
  case "$database" in
    passwd) kind=UID ;;
    group) kind=GID ;;
    *) die "unsupported identity database: $database" ;;
  esac
  # Capture enumeration before scanning: process substitution would hide a
  # failing getent behind a successful loop. Never print passwd/group records.
  records="$(getent "$database")" || die "cannot enumerate $database identities; refusing setup"
  while IFS=: read -r name password number rest; do
    [[ -n "$name" && "$number" =~ ^[0-9]+$ ]] \
      || die "invalid $database enumeration; refusing setup"
    # UID and GID are distinct namespaces. Compare only within this database,
    # numerically (including leading zeros), never UID against GID.
    if ((10#$number == 10#$expected)); then
      [[ "$name" == "$owner" ]] \
        || die "numeric $kind alias: '$owner' and '$name' share $kind $expected; refusing setup"
      found=1
    fi
  done <<<"$records"
  [[ "$found" == 1 ]] || die "cannot confirm '$owner' in $database enumeration; refusing setup"
}
validate_config_group_preflight() {
  local gid reserved reserved_gid
  CONFIG_GROUP_GID=''
  if gid="$(group_gid "$CONFIG_GROUP_NAME" 2>/dev/null)"; then
    validate_id GID "$gid" "shared configuration group '$CONFIG_GROUP_NAME'"
    validate_unique_numeric_id group "$CONFIG_GROUP_NAME" "$gid"
    CONFIG_GROUP_GID="$gid"
    for reserved in root video render; do
      if reserved_gid="$(group_gid "$reserved" 2>/dev/null)" && [[ "$gid" == "$reserved_gid" ]]; then
        die "shared configuration group '$CONFIG_GROUP_NAME' aliases reserved group '$reserved' by numeric GID"
      fi
    done
  elif getent passwd "$CONFIG_GROUP_NAME" >/dev/null 2>&1; then
    die "cannot create shared configuration group '$CONFIG_GROUP_NAME': a user with that name already exists"
  fi
}
validate_role_group() {
  local name="$1" gid="$2" reserved reserved_gid
  case "$name" in
    root|astrumweaver-config|video|render) die "role-private primary group '$name' is reserved" ;;
  esac
  validate_id GID "$gid" "role-private group '$name'"
  validate_unique_numeric_id group "$name" "$gid"
  [[ -z "$CONFIG_GROUP_GID" || "$gid" != "$CONFIG_GROUP_GID" ]] \
    || die "role-private group '$name' aliases shared configuration group by numeric GID"
  for reserved in root video render; do
    if reserved_gid="$(group_gid "$reserved" 2>/dev/null)" && [[ "$gid" == "$reserved_gid" ]]; then
      die "role-private group '$name' aliases reserved group '$reserved' by numeric GID"
    fi
  done
}
validate_existing_live_service_user() {
  local user="$1" ids uid primary gid
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"
  require_cmd getent
  ids="$(passwd_ids "$user" 2>/dev/null || true)"
  [[ -n "$ids" ]] || return 0
  IFS=: read -r uid primary <<<"$ids"
  validate_id UID "$uid" "existing service account '$user'"
  validate_unique_numeric_id passwd "$user" "$uid"
  validate_id GID "$primary" "existing service account '$user' primary group"
  gid="$(group_gid "$user" 2>/dev/null || true)"
  [[ -n "$gid" ]] || die "existing service account '$user' has no same-name service group '$user'; refusing setup"
  validate_id GID "$gid" "same-name service group '$user'"
  [[ "$primary" == "$gid" ]] || die "existing service account '$user' primary GID=$primary does not match same-name group '$user' GID=$gid"
  [[ -z "$CONFIG_GROUP_GID" || "$primary" != "$CONFIG_GROUP_GID" ]] \
    || die "existing service account '$user' uses shared configuration group '$CONFIG_GROUP_NAME' as its primary group"
  validate_role_group "$user" "$gid"
}
group_has_other_named_member() {
  local group="$1" owner="$2" member
  while IFS= read -r member; do
    [[ -n "$member" && "$member" != "$owner" ]] && return 0
  done < <(group_members "$group" 2>/dev/null | tr ',' '\n' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
  return 1
}
group_has_other_primary_member() {
  local group="$1" gid_target="$2" name password uid gid rest
  while IFS=: read -r name password uid gid rest; do
    [[ "$gid" == "$gid_target" && "$name" != "$group" ]] && return 0
  done < <(getent passwd || true)
  return 1
}
user_is_member_of_gid() {
  local user="$1" target="$2" groups
  groups="$(id -G "$user" 2>/dev/null)" || die "cannot determine group membership for '$user'; refusing setup"
  printf '%s\n' "$groups" | tr '[:space:]' '\n' | grep -Fqx -- "$target"
}
validate_role_group_members() {
  local user="$1" gid="$2"
  if group_has_other_primary_member "$user" "$gid" || group_has_other_named_member "$user" "$user"; then
    die "role-private group '$user' contains another account; supplementary membership cannot include a peer role group"
  fi
}
validate_live_service_accounts() {
  local role="$1" user="$2" peer_ids='' peer_uid='' peer_gid='' current_ids='' current_uid='' current_gid='' current_exists=0
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"
  require_cmd getent; require_cmd id; require_cmd groupadd; require_cmd useradd; require_cmd usermod
  validate_config_group_preflight

  current_ids="$(passwd_ids "$user" 2>/dev/null || true)"
  if [[ -n "$current_ids" ]]; then
    current_exists=1
    IFS=: read -r current_uid current_gid <<<"$current_ids"
    validate_existing_live_service_user "$user" "$role"
    validate_role_group_members "$user" "$current_gid"
  elif current_gid="$(group_gid "$user" 2>/dev/null)"; then
    validate_role_group "$user" "$current_gid"
    if [[ -n "$(group_members "$user" 2>/dev/null)" ]] || group_has_other_primary_member "$user" "$current_gid"; then
      die "pre-provisioned role-private group '$user' must be empty when its service user is absent"
    fi
  fi

  if [[ -n "$ROLE_PEER_USER" ]]; then
    peer_ids="$(passwd_ids "$ROLE_PEER_USER" 2>/dev/null || true)"
    [[ -n "$peer_ids" ]] || legacy_service_migration "$ROLE_PEER_UNIT" "installed peer unit User=$ROLE_PEER_USER has no corresponding live service account"
    IFS=: read -r peer_uid peer_gid <<<"$peer_ids"
    validate_existing_live_service_user "$ROLE_PEER_USER"
    peer_gid="$(group_gid "$ROLE_PEER_GROUP" 2>/dev/null || true)"
    [[ -n "$peer_gid" ]] || die "peer service account '$ROLE_PEER_USER' has no same-name service group '$ROLE_PEER_GROUP'"
    validate_role_group_members "$ROLE_PEER_USER" "$peer_gid"
  fi

  if [[ -n "$ROLE_PEER_USER" && -n "$current_gid" && "$current_gid" == "$peer_gid" ]]; then
    die "cross-role numeric GID alias: '$user' and '$ROLE_PEER_USER' private groups share GID $current_gid"
  fi
  if [[ "$current_exists" == 1 && -n "$ROLE_PEER_USER" ]]; then
    if [[ "$current_uid" == "$peer_uid" ]]; then
      die "cross-role numeric UID alias: '$user' and '$ROLE_PEER_USER' share UID $current_uid"
    fi
    if user_is_member_of_gid "$user" "$peer_gid"; then
      die "service account '$user' is a member of peer private group '$ROLE_PEER_GROUP'; supplementary membership cannot include a peer role group"
    fi
    if user_is_member_of_gid "$ROLE_PEER_USER" "$current_gid"; then
      die "peer service account '$ROLE_PEER_USER' is a member of role-private group '$user'; supplementary membership cannot include a peer role group"
    fi
  fi
}
ensure_live_service_user() {
  local user="$1" state_dir="$2" role="${3:-service}"
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"
  reject_symlink_path "$state_dir"
  validate_existing_live_service_user "$user" "$role"
  if getent passwd "$user" >/dev/null 2>&1; then return 0; fi
  require_cmd useradd
  if getent group "$user" >/dev/null 2>&1; then
    useradd --system --gid "$user" --home-dir "$state_dir" --create-home --shell /usr/sbin/nologin "$user"
  else
    useradd --system --user-group --home-dir "$state_dir" --create-home --shell /usr/sbin/nologin "$user"
  fi
  validate_existing_live_service_user "$user" "$role"
}
ensure_service_config_membership() {
  local user="$1"
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"
  require_cmd groupadd; require_cmd usermod
  if ! getent group "$CONFIG_GROUP_NAME" >/dev/null 2>&1; then groupadd --system "$CONFIG_GROUP_NAME"; fi
  validate_config_group_preflight
  usermod --append --groups "$CONFIG_GROUP_NAME" "$user"
  user_is_member_of_gid "$user" "$CONFIG_GROUP_GID" || die "service account '$user' is not a supplementary member of '$CONFIG_GROUP_NAME' after setup"
}
reconcile_service_directories() {
  local root="$1" user="$2" config_group="$3" etc_dir="$4" state_dir="$5"
  reject_symlink_path "$etc_dir" "$state_dir"
  [[ -d "$etc_dir" ]] || die "service configuration directory is missing: $etc_dir"
  [[ -d "$state_dir" ]] || die "service state directory is missing: $state_dir"
  if [[ "$root" == / ]]; then chown "root:$config_group" "$etc_dir"; chown "$user:$user" "$state_dir"; fi
  chmod 0710 "$etc_dir"; chmod 0750 "$state_dir"
}
reconcile_service_file() {
  local root="$1" user="$2" path="$3"
  reject_symlink_path "$path"
  [[ -e "$path" ]] || return 0
  [[ -f "$path" ]] || die "service file is not a regular file: $path"
  [[ "$root" != / ]] || chown "root:$user" "$path"
  chmod 0640 "$path"
}
resolve_executable() {
  local requested="$1" root="$2" default_command="$3" resolved
  if [[ -n "$requested" ]]; then
    [[ "$requested" == /* ]] || die "--executable must be an absolute path"
    if [[ "$root" == / ]]; then [[ -x "$requested" ]] || die "executable is not executable: $requested"; fi
    printf '%s\n' "$requested"; return 0
  fi
  [[ "$root" == / ]] || die "--executable is required for staged --root installs"
  resolved="$(command -v "$default_command" || true)"
  [[ -n "$resolved" && -x "$resolved" ]] || die "packaged executable not found on PATH: $default_command"
  printf '%s\n' "$resolved"
}
systemd_reload_and_maybe_start() {
  local service="$1" start="$2"
  require_cmd systemctl; systemctl daemon-reload
  [[ "$start" != 1 ]] || systemctl enable --now "$service"
}
