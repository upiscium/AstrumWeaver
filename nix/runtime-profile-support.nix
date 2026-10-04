{ lib, coreutils, nix, runtimeLlamaCpp, source, writeShellApplication }:

let
  profile = "/nix/var/nix/profiles/astrumweaver-runtime-llama-cpp";
  expectedExecutable = "${runtimeLlamaCpp}/bin/llama-server";
  candidateRef = "path:${source}";
  registryRef = "astrumweaver-runtime-candidate";
in
writeShellApplication {
  name = "astrumweaver-runtime-profile";
  text = ''
    profile=${lib.escapeShellArg profile}
    expected_executable=${lib.escapeShellArg expectedExecutable}
    candidate_ref=${lib.escapeShellArg candidateRef}
    registry_ref=${lib.escapeShellArg registryRef}
    nix_bin=${lib.escapeShellArg "${nix}/bin/nix"}
    readlink_bin=${lib.escapeShellArg "${coreutils}/bin/readlink"}

    usage() {
      cat >&2 <<'EOF'
usage:
  astrumweaver-runtime-profile verify package llama-cpp
  astrumweaver-runtime-profile ensure llama-cpp
  astrumweaver-runtime-profile upgrade llama-cpp
  astrumweaver-runtime-profile rollback llama-cpp
  astrumweaver-runtime-profile status llama-cpp
EOF
      exit 2
    }

    require_llama_cpp() {
      if [ "${1-}" != "llama-cpp" ]; then
        echo "astrumweaver-runtime-profile: unsupported runtime package: ${1-<missing>}" >&2
        exit 2
      fi
    }

    resolved_executable() {
      if [ ! -x "$profile/bin/llama-server" ]; then
        return 1
      fi
      "$readlink_bin" -f "$profile/bin/llama-server"
    }

    verify_llama_cpp() {
      actual="$(resolved_executable 2>/dev/null || true)"
      [ -n "$actual" ] && [ "$actual" = "$expected_executable" ]
    }

    status_llama_cpp() {
      if verify_llama_cpp; then
        echo "satisfied: $profile/bin/llama-server -> $expected_executable"
        return 0
      fi
      if [ -e "$profile/bin/llama-server" ] || [ -L "$profile/bin/llama-server" ]; then
        actual="$(resolved_executable 2>/dev/null || true)"
        echo "candidate-mismatch: $profile/bin/llama-server -> ${actual:-unresolved}" >&2
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
        echo "astrumweaver-runtime-profile: Nix completed but the reviewed llama-server binding is not active" >&2
        exit 1
      fi
      echo "runtime profile now matches this AstrumWeaver candidate"
    }

    case "${1-}" in
      verify)
        [ "${2-}" = "package" ] || usage
        require_llama_cpp "${3-}"
        verify_llama_cpp
        ;;
      ensure)
        require_llama_cpp "${2-}"
        install_or_upgrade_llama_cpp
        ;;
      upgrade)
        require_llama_cpp "${2-}"
        install_or_upgrade_llama_cpp
        ;;
      rollback)
        require_llama_cpp "${2-}"
        "$nix_bin" profile rollback --profile "$profile"
        status_llama_cpp || true
        ;;
      status)
        require_llama_cpp "${2-}"
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
