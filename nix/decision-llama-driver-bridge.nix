# Isolated host libcuda loader for a Nix-built llama-server on a non-NixOS LXC.
# Only the driver library is added to the search path; host glibc and
# libstdc++ are never inserted into the Nix runtime's library resolution.
{ writeShellApplication, coreutils, core }:

writeShellApplication {
  name = "astrumweaver-llama-server";
  runtimeInputs = [ coreutils ];
  text = ''
    set -euo pipefail

    driver="''${ASTRUMWEAVER_LIBCUDA_SO:-}"
    if [[ -z "$driver" ]]; then
      for candidate in \
        /run/opengl-driver/lib/libcuda.so.1 \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
        /lib/x86_64-linux-gnu/libcuda.so.1; do
        if [[ -f "$candidate" ]]; then
          driver="$candidate"
          break
        fi
      done
    fi
    if [[ -z "$driver" || ! -f "$driver" ]]; then
      echo "astrumweaver-llama-server: cuda_driver_library_unavailable" >&2
      exit 65
    fi
    driver="$(readlink -f -- "$driver")"
    if [[ "$driver" == */stubs/* ]]; then
      echo "astrumweaver-llama-server: cuda_stub_is_not_a_driver" >&2
      exit 65
    fi

    private="$(mktemp -d -t astrumweaver-cuda-bridge.XXXXXXXX)"
    chmod 700 "$private"
    ln -s -- "$driver" "$private/libcuda.so.1"
    export LD_LIBRARY_PATH="$private"
    # The package's ELF binaries keep the pinned Nix RUNPATH dependencies.
    # Do not inherit the calling host's potentially incompatible LD_LIBRARY_PATH.
    child=""
    cleanup() {
      status=$?
      trap - EXIT TERM INT HUP
      if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
      fi
      rm -rf -- "$private"
      exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    trap 'exit 129' HUP
    "${core}/bin/llama-server" "$@" &
    child=$!
    wait "$child"
  '';
}
