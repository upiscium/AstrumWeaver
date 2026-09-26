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

ensure_live_service_user() {
  local user="$1"
  [[ "$EUID" -eq 0 ]] || die "live setup must run as root"
  if id "$user" >/dev/null 2>&1; then
    return 0
  fi
  require_cmd useradd
  useradd --system --home-dir /var/lib/astrumweaver --create-home \
    --shell /usr/sbin/nologin "$user"
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
