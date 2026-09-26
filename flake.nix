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
      runtimeIsolationPreflight =
        runtimeModuleSmoke.config.systemd.services.astrumweaver-worker-gpu-isolation-preflight.serviceConfig.ExecStart;
    in
    {
      packages.${system} = {
        inherit astrumweaver control worker integration;
        default = astrumweaver;
      };

      nixosModules = {
        control = import ./nix/modules/control.nix;
        worker = import ./nix/modules/worker.nix;
        default = import ./nix/modules/default.nix;
      };

      nixosConfigurations.smoke = moduleSmoke;

      checks.${system} = {
        inherit astrumweaver control worker integration;

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
          test "$runtimeDevicePolicy" = closed
          printf "%s" "$runtimeDeviceAllowText" | grep -q '/dev/nvidia7 rw'
          printf "%s" "$runtimeDeviceAllowText" | grep -q '/dev/nvidiactl rw'
          printf "%s" "$runtimeEnvironmentText" | grep -q 'CUDA_VISIBLE_DEVICES=GPU-example-smoke'
          printf "%s" "$runtimeIsolationPreflight" | grep -q 'gpu-device-map verify'
          printf "%s\n%s\n%s\n%s\n%s\n" \
            "$runtimeWorkerExec" "$runtimeWorkerPath" "$workerConfig" \
            "$runtimeDevicePolicy" "$runtimeIsolationPreflight" > "$out"
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
