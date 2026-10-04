{ lib, nix, python312, source, writeShellApplication }:

let
  profile = "/nix/var/nix/profiles/astrumweaver-runtime-llama-cpp";
  candidateRef = "path:${source}";
  registryRef = "astrumweaver-runtime-candidate";
in
writeShellApplication {
  name = "astrumweaver-runtime-profile";
  text = ''
    profile=${lib.escapeShellArg profile}
    candidate_ref=${lib.escapeShellArg candidateRef}
    registry_ref=${lib.escapeShellArg registryRef}
    nix_bin=${lib.escapeShellArg "${nix}/bin/nix"}
    python_bin=${lib.escapeShellArg "${python312}/bin/python3"}

    usage() {
      cat >&2 <<'EOF'
usage:
  astrumweaver-runtime-profile verify package llama-cpp
  astrumweaver-runtime-profile ensure llama-cpp
  astrumweaver-runtime-profile upgrade llama-cpp
  astrumweaver-runtime-profile rollback llama-cpp
  astrumweaver-runtime-profile status llama-cpp
EOF
      exit "''${1:-2}"
    }

    require_llama_cpp() {
      if [ "''${1-}" != "llama-cpp" ]; then
        echo "astrumweaver-runtime-profile: unsupported runtime package: ''${1-<missing>}" >&2
        exit 2
      fi
    }

    verify_llama_cpp() {
      if [ ! -x "$profile/bin/llama-server" ]; then
        return 1
      fi
      "$python_bin" - "$nix_bin" "$profile" "$candidate_ref" "$registry_ref" <<'PY'
import json
import subprocess
import sys

nix_bin, profile, candidate_ref, registry_ref = sys.argv[1:]

def run_json(args):
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise SystemExit(1)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise SystemExit(1)

profile_data = run_json(
    [nix_bin, "profile", "list", "--profile", profile, "--json"]
)
candidate_data = run_json(
    [nix_bin, "flake", "metadata", "--json", candidate_ref]
)
candidate_url = candidate_data.get("url") or candidate_data.get("lockedUrl")
if not isinstance(candidate_url, str) or not candidate_url:
    raise SystemExit(1)

elements = profile_data.get("elements", {})
if isinstance(elements, dict):
    values = list(elements.values())
elif isinstance(elements, list):
    values = elements
else:
    raise SystemExit(1)

expected_original = "flake:" + registry_ref
expected_attr = "packages.x86_64-linux.runtime-llama-cpp"
matches = [
    element
    for element in values
    if isinstance(element, dict)
    and element.get("active", True)
    and element.get("originalUrl") == expected_original
    and element.get("attrPath") == expected_attr
    and (element.get("url") or element.get("uri")) == candidate_url
]
raise SystemExit(0 if len(matches) == 1 and len(values) == 1 else 1)
PY
    }

    status_llama_cpp() {
      if verify_llama_cpp; then
        echo "satisfied: $profile is bound to this AstrumWeaver candidate"
        return 0
      fi
      if [ -x "$profile/bin/llama-server" ]; then
        echo "candidate-mismatch: $profile does not match this AstrumWeaver candidate" >&2
      else
        echo "missing: $profile/bin/llama-server" >&2
      fi
      return 1
    }

    install_or_upgrade_llama_cpp() {
      if verify_llama_cpp; then
        echo "runtime profile already matches this AstrumWeaver candidate"
        return 0
      fi

      if [ -e "$profile" ] || [ -L "$profile" ]; then
        "$nix_bin" profile upgrade \
          --profile "$profile" \
          --all \
          --override-flake "$registry_ref" "$candidate_ref"
      else
        "$nix_bin" profile add \
          --profile "$profile" \
          --override-flake "$registry_ref" "$candidate_ref" \
          "$registry_ref#runtime-llama-cpp"
      fi

      if ! verify_llama_cpp; then
        echo "astrumweaver-runtime-profile: Nix completed but the reviewed candidate binding is not active" >&2
        exit 1
      fi
      echo "runtime profile now matches this AstrumWeaver candidate"
    }

    case "''${1-}" in
      -h|--help)
        usage 0
        ;;
      verify)
        [ "''${2-}" = "package" ] || usage
        require_llama_cpp "''${3-}"
        verify_llama_cpp
        ;;
      ensure)
        require_llama_cpp "''${2-}"
        install_or_upgrade_llama_cpp
        ;;
      upgrade)
        require_llama_cpp "''${2-}"
        install_or_upgrade_llama_cpp
        ;;
      rollback)
        require_llama_cpp "''${2-}"
        "$nix_bin" profile rollback --profile "$profile"
        status_llama_cpp || true
        ;;
      status)
        require_llama_cpp "''${2-}"
        status_llama_cpp
        ;;
      *)
        usage
        ;;
    esac
  '';
  meta = {
    description = "AstrumWeaver generic-systemd RuntimeBackend Nix profile manager";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
