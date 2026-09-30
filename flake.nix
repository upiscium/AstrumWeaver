{
  description = "AstrumWeaver deployment-agnostic heterogeneous compute fabric";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      astrumweaver = pkgs.callPackage ./nix/package.nix { };
      integration = pkgs.callPackage ./nix/integration-package.nix { };
      control = pkgs.callPackage ./nix/control-support.nix {
        inherit astrumweaver integration;
      };
      worker = pkgs.callPackage ./nix/worker-support.nix {
        inherit astrumweaver integration;
      };
      installer = pkgs.callPackage ./nix/installer-support.nix {
        inherit astrumweaver integration;
      };
      fakeNvidia = pkgs.writeShellScriptBin "nvidia-smi" ''
        if [ "$1" = "--query-gpu=uuid" ]; then
          echo GPU-example-smoke
          exit 0
        fi
        exit 0
      '';
      fakeOllama = pkgs.writeShellScriptBin "ollama" ''
        exit 0
      '';

      moduleSmoke = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          ({ ... }: {
            system.stateVersion = "26.05";
            boot.isContainer = true;
            fileSystems."/" = {
              device = "none";
              fsType = "tmpfs";
            };

            services.astrumweaver.control = {
              enable = true;
              package = control;
              settings.control = {
                host = "127.0.0.1";
                port = 9000;
              };
            };

            services.astrumweaver.worker = {
              enable = true;
              package = worker;
              workerId = "smoke-worker";
              workerClass = "modern-single";
              controlUrl = "http://127.0.0.1:9000";
              capabilities = [ "debug.echo" ];
              gpuUuids = [ "GPU-example-smoke" ];
              totalVramMb = 16384;
              maxSingleGpuVramMb = 16384;
              executorFactory = "astrumweaver.executors.structured_echo:create_executor";
              nvidiaSmiPackage = fakeNvidia;
              borrowable.enable = true;
            };
          })
        ];
      };
      runtimeModuleSmoke = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          ({ ... }: {
            system.stateVersion = "26.05";
            boot.isContainer = true;
            fileSystems."/" = {
              device = "none";
              fsType = "tmpfs";
            };

            services.astrumweaver.worker = {
              enable = true;
              package = worker;
              workerId = "runtime-smoke-worker";
              workerClass = "modern-single";
              controlUrl = "http://127.0.0.1:9000";
              capabilities = [ "llm.chat" "text.generate" ];
              gpuUuids = [ "GPU-example-smoke" ];
              accelerators = [
                {
                  uuid = "GPU-example-smoke";
                  memory_mb = 16384;
                  compute_capability = "8.6";
                  device_class = "NVIDIA Test GPU";
                }
              ];
              totalVramMb = 16384;
              maxSingleGpuVramMb = 16384;
              nvidiaSmiPackage = fakeNvidia;
              gpuIsolation = {
                enable = true;
                deviceMap."GPU-example-smoke" = "/dev/nvidia7";
                auxiliaryDeviceNodes = [ "/dev/nvidiactl" ];
              };

              runtime = {
                enable = true;
                provider = "ollama";
                packages = [ fakeOllama ];
                modelRef = "qwen3:8b";
                modelFormat = "ollama";
                modelTopology = "dense";
                residencyPolicy = "prefer_vram";
                providerConfig.keep_alive = "5m";
              };
            };
          })
        ];
      };
      gpuIsolationNoGpuSmoke = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          ({ ... }: {
            system.stateVersion = "26.05";
            boot.isContainer = true;
            services.astrumweaver.worker = {
              enable = true;
              gpuIsolation.enable = true;
            };
          })
        ];
      };
      gpuIsolationNoGpuRejected = builtins.any
        (assertion:
          !assertion.assertion
          && assertion.message == "gpuIsolation.enable requires at least one selected GPU UUID.")
        gpuIsolationNoGpuSmoke.config.assertions;
      borrowableModePackages = builtins.filter (
        package:
          nixpkgs.lib.getName package == "astrumweaver-gpu-mode"
      ) moduleSmoke.config.environment.systemPackages;
      borrowableMode = builtins.head borrowableModePackages;
      runtimeWorkerExec =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStart;
      runtimeWorkerPath = nixpkgs.lib.makeBinPath
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.path;
      runtimeDevicePolicy =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.DevicePolicy;
      runtimeDeviceAllow =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.DeviceAllow;
      runtimeEnvironment =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.Environment;
      runtimeWorkerPreflight =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStartPre;
      runtimeIsolationPreflight =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker-gpu-isolation-preflight.script;
    in
    {
      packages.${system} = {
        inherit astrumweaver control worker installer integration;
        default = astrumweaver;
      };

      nixosModules = {
        control = import ./nix/modules/control.nix;
        worker = import ./nix/modules/worker.nix;
        default = import ./nix/modules/default.nix;
      };

      nixosConfigurations.smoke = moduleSmoke;

      checks.${system} = {
        inherit astrumweaver control worker installer integration;

        setup-permissions = import ./nix/tests/setup-permissions.nix {
          inherit pkgs integration;
        };

        installation-surface = pkgs.runCommand "astrumweaver-installation-surface" { } ''
          test -x ${installer}/bin/astrumweaver-setup-tui
          test -x ${installer}/bin/astrumweaver-control
          test -x ${installer}/bin/astrumweaver-migrate
          test -x ${installer}/bin/astrumweaver-worker
          test -x ${installer}/bin/astrumweaver-setup-control-plane
          test -x ${installer}/bin/astrumweaver-setup-gpu-worker
          test -x ${control}/bin/astrumweaver-control
          test -x ${control}/bin/astrumweaver-migrate
          test -x ${control}/bin/astrumweaver-setup-control-plane
          test -x ${worker}/bin/astrumweaver-worker
          test -x ${worker}/bin/astrumweaver-setup-gpu-worker
          test -x ${worker}/bin/astrumweaver-setup-tui
          test -x ${worker}/bin/astrumweaver-runtime-deployment-accept

          # Execute the actual packaged TUI wrapper through a profile-like
          # symlink.  The wrapper must pass that lexical bin directory into
          # Python; synthetic sys.argv[0] assignment is deliberately not used.
          TEST_ROOT="$PWD/installation-surface-test"
          mkdir -p "$TEST_ROOT/installer-generation-a" "$TEST_ROOT/installer-generation-b"
          cp -a ${installer}/. "$TEST_ROOT/installer-generation-a/"
          cp -a ${installer}/. "$TEST_ROOT/installer-generation-b/"
          ln -s "$TEST_ROOT/installer-generation-a" "$TEST_ROOT/installer-profile"
          PROFILE="$TEST_ROOT/installer-profile"
          MINIMAL_PATH=${pkgs.coreutils}/bin:${pkgs.bash}/bin
          PATH="$MINIMAL_PATH" \
            "$PROFILE/bin/astrumweaver-setup-tui" --check-packaging \
            > "$TEST_ROOT/packaging-a"
          grep -Fq "packaged tool authority: $PROFILE/bin" "$TEST_ROOT/packaging-a"
          grep -Fq "astrumweaver-worker: $PROFILE/bin/astrumweaver-worker" "$TEST_ROOT/packaging-a"
          grep -Fq "astrumweaver-control: $PROFILE/bin/astrumweaver-control" "$TEST_ROOT/packaging-a"
          (
            cd "$TEST_ROOT"
            PATH="$PROFILE/bin:$MINIMAL_PATH" astrumweaver-setup-tui --check-packaging \
              > "$TEST_ROOT/packaging-bare"
          )
          grep -Fq "packaged tool authority: $PROFILE/bin" "$TEST_ROOT/packaging-bare"

          mkdir -p "$TEST_ROOT/input" "$TEST_ROOT/root"
          cat > "$TEST_ROOT/input/control.toml" <<'EOF'
[control]
host = "127.0.0.1"
port = 9000
EOF
          cat > "$TEST_ROOT/input/worker.toml" <<'EOF'
[worker]
id = "profile-convergence-worker"
class = "modern-single"
gpu_uuids = ["GPU-profile-convergence"]
EOF
          "$PROFILE/bin/astrumweaver-setup-control-plane" \
            --config "$TEST_ROOT/input/control.toml" \
            --executable "$PROFILE/bin/astrumweaver-control" \
            --root "$TEST_ROOT/root"
          "$PROFILE/bin/astrumweaver-setup-gpu-worker" \
            --config "$TEST_ROOT/input/worker.toml" \
            --executable "$PROFILE/bin/astrumweaver-worker" \
            --root "$TEST_ROOT/root"
          cp "$TEST_ROOT/root/etc/systemd/system/astrumweaver-control.service" "$TEST_ROOT/control-a"
          cp "$TEST_ROOT/root/etc/systemd/system/astrumweaver-worker.service" "$TEST_ROOT/worker-a"
          cp "$TEST_ROOT/root/usr/local/libexec/astrumweaver/gpu-device-map" "$TEST_ROOT/gpu-map-a"
          grep -Fq "ExecStart=$PROFILE/bin/astrumweaver-control " "$TEST_ROOT/control-a"
          grep -Fq "ExecStart=$PROFILE/bin/astrumweaver-worker " "$TEST_ROOT/worker-a"
          ! grep -Fq "/nix/store/" "$TEST_ROOT/control-a"
          ! grep -Fq "/nix/store/" "$TEST_ROOT/worker-a"
          ! grep -Fq "/nix/store/" "$TEST_ROOT/gpu-map-a"

          rm -f "$PROFILE"
          ln -s "$TEST_ROOT/installer-generation-b" "$PROFILE"
          PATH="$MINIMAL_PATH" \
            "$PROFILE/bin/astrumweaver-setup-tui" --check-packaging \
            > "$TEST_ROOT/packaging-b"
          grep -Fq "packaged tool authority: $PROFILE/bin" "$TEST_ROOT/packaging-b"
          "$PROFILE/bin/astrumweaver-setup-control-plane" \
            --config "$TEST_ROOT/input/control.toml" \
            --executable "$PROFILE/bin/astrumweaver-control" \
            --root "$TEST_ROOT/root"
          "$PROFILE/bin/astrumweaver-setup-gpu-worker" \
            --config "$TEST_ROOT/input/worker.toml" \
            --executable "$PROFILE/bin/astrumweaver-worker" \
            --root "$TEST_ROOT/root"
          cmp "$TEST_ROOT/control-a" "$TEST_ROOT/root/etc/systemd/system/astrumweaver-control.service"
          cmp "$TEST_ROOT/worker-a" "$TEST_ROOT/root/etc/systemd/system/astrumweaver-worker.service"
          cmp "$TEST_ROOT/gpu-map-a" "$TEST_ROOT/root/usr/local/libexec/astrumweaver/gpu-device-map"

          if PATH="$MINIMAL_PATH" "$PROFILE/bin/astrumweaver-runtime-deployment-accept" \
            --gpu-uuid GPU-profile-convergence \
            --revision deadbeef12345678 \
            --deployment-path systemd \
            --worker-config "$TEST_ROOT/missing-worker.toml" \
            --systemctl ${pkgs.coreutils}/bin/false \
            --nvidia-smi ${pkgs.coreutils}/bin/false \
            > "$TEST_ROOT/runtime-acceptance-resolution" 2>&1; then
            exit 1
          fi
          ! grep -Fq "reviewed packaged GPU mapper is unavailable" "$TEST_ROOT/runtime-acceptance-resolution"
          test "$(readlink "$PROFILE")" = "$TEST_ROOT/installer-generation-b"
          touch "$out"
        '';
        hardware-accept-cli = pkgs.runCommand "astrumweaver-hardware-accept-cli" { } ''
          test -x ${worker}/bin/astrumweaver-hardware-accept
          ${worker}/bin/astrumweaver-hardware-accept --help >/dev/null
          touch "$out"
        '';
        runtime-deployment-accept-cli = pkgs.runCommand "astrumweaver-runtime-deployment-accept-cli" { } ''
          test -x ${worker}/bin/astrumweaver-runtime-deployment-accept
          ${worker}/bin/astrumweaver-runtime-deployment-accept --help >/dev/null
          touch "$out"
        '';
        runtime-module-eval = pkgs.runCommand "astrumweaver-runtime-module-eval" {
          inherit
            runtimeWorkerExec
            runtimeWorkerPath
            runtimeDevicePolicy
            runtimeWorkerPreflight
            runtimeIsolationPreflight
            ;
          runtimeDeviceAllowText = nixpkgs.lib.concatStringsSep "\n" runtimeDeviceAllow;
          runtimeEnvironmentText = nixpkgs.lib.concatStringsSep "\n" runtimeEnvironment;
        } ''
          test -n "$runtimeWorkerExec"
          printf "%s" "$runtimeWorkerExec" | grep -q astrumweaver-worker
          printf "%s" "$runtimeWorkerPath" | grep -q ollama
          workerConfig="$(printf "%s" "$runtimeWorkerExec" | sed -n 's/.*--config \([^ ]*\).*/\1/p')"
          test -n "$workerConfig"
          test -f "$workerConfig"
          grep -q '\[runtime\]' "$workerConfig"
          grep -q 'manifest' "$workerConfig"
          grep -q 'gpu_isolation_visible_map' "$workerConfig"
          test "$runtimeDevicePolicy" = closed
          printf "%s" "$runtimeDeviceAllowText" | grep -q '/dev/nvidia7 rw'
          printf "%s" "$runtimeDeviceAllowText" | grep -q '/dev/nvidiactl rw'
          printf "%s" "$runtimeEnvironmentText" | grep -q 'CUDA_VISIBLE_DEVICES=GPU-example-smoke'
          printf "%s" "$runtimeEnvironmentText" | grep -q 'ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP=/run/astrumweaver-worker-gpu-isolation/gpu-visible-map'
          printf "%s" "$runtimeWorkerPreflight" | grep -q 'astrumweaver-gpu-device-map'
          printf "%s" "$runtimeWorkerPreflight" | grep -q 'probe-access'
          printf "%s" "$runtimeWorkerPreflight" | grep -q '/run/astrumweaver-worker-gpu-isolation/gpu-visible-map'
          printf "%s" "$runtimeIsolationPreflight" | grep -q 'snapshot-visible'
          printf "%s" "$runtimeIsolationPreflight" | grep -q 'gpu-device-map'
          printf "%s" "$runtimeIsolationPreflight" | grep -q 'verify'
          printf "%s\n%s\n%s\n%s\n%s\n" \
            "$runtimeWorkerExec" "$runtimeWorkerPath" "$workerConfig" \
            "$runtimeDevicePolicy" "$runtimeWorkerPreflight" \
            "$runtimeIsolationPreflight" > "$out"
        '';
        gpu-isolation-requires-gpu =
          assert gpuIsolationNoGpuRejected;
          pkgs.runCommand "astrumweaver-gpu-isolation-requires-gpu" { } ''
            touch "$out"
          '';
        module-eval = pkgs.runCommand "astrumweaver-module-eval" {
          controlExec = moduleSmoke.config.systemd.services.astrumweaver-control.serviceConfig.ExecStart;
          workerExec = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStart;
          workerPreflight = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStartPre;
          modeTool = borrowableMode;
        } ''
          test -n "$controlExec"
          test -n "$workerExec"
          test -n "$workerPreflight"
          test -x "$modeTool/bin/astrumweaver-gpu-mode"
          printf "%s" "$controlExec" | grep -q astrumweaver-control
          printf "%s" "$workerExec" | grep -q astrumweaver-worker
          printf "%s\n%s\n%s\n%s\n" "$controlExec" "$workerExec" "$workerPreflight" "$modeTool" > "$out"
        '';
      };
    };
}

          printf "%s" "$runtimeIsolationPreflight" | grep -q 'snapshot-visible'
          printf "%s" "$runtimeIsolationPreflight" | grep -q 'gpu-device-map.*verify'
          printf "%s\n%s\n%s\n%s\n%s\n" \
            "$runtimeWorkerExec" "$runtimeWorkerPath" "$workerConfig" \
            "$runtimeDevicePolicy" "$runtimeWorkerPreflight" \
            "$runtimeIsolationPreflight" > "$out"
        '';
        gpu-isolation-requires-gpu =
          assert gpuIsolationNoGpuRejected;
          pkgs.runCommand "astrumweaver-gpu-isolation-requires-gpu" { } ''
            touch "$out"
          '';
        module-eval = pkgs.runCommand "astrumweaver-module-eval" {
          controlExec = moduleSmoke.config.systemd.services.astrumweaver-control.serviceConfig.ExecStart;
          workerExec = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStart;
          workerPreflight = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStartPre;
          modeTool = borrowableMode;
        } ''
          test -n "$controlExec"
          test -n "$workerExec"
          test -n "$workerPreflight"
          test -x "$modeTool/bin/astrumweaver-gpu-mode"
          printf "%s" "$controlExec" | grep -q astrumweaver-control
          printf "%s" "$workerExec" | grep -q astrumweaver-worker
          printf "%s\n%s\n%s\n%s\n" "$controlExec" "$workerExec" "$workerPreflight" "$modeTool" > "$out"
        '';
      };
    };
}
