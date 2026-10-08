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
    # The managed Worker carries Control authority in its environment.
    # Inference must not inherit Worker/Client tokens or ambient cloud secrets.
    # The Nix loader uses its own store RUNPATH, plus only this host driver shim.
    runtime_env=( "LD_LIBRARY_PATH=$private" "PATH=$PATH" )
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
      runtime_env+=( "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" )
    fi
    if [[ -v CUDA_DEVICE_ORDER ]]; then
      runtime_env+=( "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER" )
    fi
    if [[ -v HOME ]]; then
      runtime_env+=( "HOME=$HOME" )
    fi
    if [[ -v TMPDIR ]]; then
      runtime_env+=( "TMPDIR=$TMPDIR" )
    fi
    if [[ -v LANG ]]; then
      runtime_env+=( "LANG=$LANG" )
    fi
    if [[ -v LC_ALL ]]; then
      runtime_env+=( "LC_ALL=$LC_ALL" )
    fi
    child=""
    cleanup() {
      status=$?
      trap - EXIT TERM INT HUP
      if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
        # Never let a non-cooperative runtime prevent driver-shim cleanup
        # indefinitely when systemd/Worker requests shutdown.
        for ((attempt = 0; attempt < 80; attempt++)); do
          if ! kill -0 "$child" 2>/dev/null; then
            break
          fi
          sleep 0.1
        done
        if kill -0 "$child" 2>/dev/null; then
          kill -KILL "$child" 2>/dev/null || true
        fi
        wait "$child" 2>/dev/null || true
      fi
      rm -rf -- "$private"
      exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    trap 'exit 129' HUP
    env -i "''${runtime_env[@]}" "${core}/bin/llama-server" "$@" &
    child=$!
    wait "$child"
  '';
}
