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
              gpuUuids = [ "GPU-example-smoke" ];
              nvidiaSmiPackage = fakeNvidia;
              settings = {
                worker = {
                  id = "smoke-worker";
                  class = "modern-single";
                  control_url = "http://127.0.0.1:9000";
                  gpu_uuids = [ "GPU-example-smoke" ];
                  gpu_count = 1;
                  total_vram_mb = 16384;
                  max_single_gpu_vram_mb = 16384;
                  capabilities = [ "debug.echo" ];
                };
                executor = {
                  factory = "astrumweaver.executors.structured_echo:create_executor";
                  settings = { };
                };
              };
            };
          })
        ];
      };
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

        module-eval = pkgs.runCommand "astrumweaver-module-eval" {
          controlExec = moduleSmoke.config.systemd.services.astrumweaver-control.serviceConfig.ExecStart;
          workerExec = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStart;
          workerPreflight = moduleSmoke.config.systemd.services.astrumweaver-worker.serviceConfig.ExecStartPre;
        } ''
          test -n "$controlExec"
          test -n "$workerExec"
          test -n "$workerPreflight"
          printf "%s" "$controlExec" | grep -q astrumweaver-control
          printf "%s" "$workerExec" | grep -q astrumweaver-worker
          printf "%s\n%s\n%s\n" "$controlExec" "$workerExec" "$workerPreflight" > "$out"
        '';
      };
    };
}
