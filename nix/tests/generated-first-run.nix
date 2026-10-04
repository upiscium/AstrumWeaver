# Evaluates the exact files emitted by tools/check_generated_nixos.py. Do not
# replace them with handwritten modules: their lexical scope is under test.
{ source, manifest }:
let
  flake = builtins.getFlake ("path:" + source);
  lib = flake.inputs.nixpkgs.lib;
  system = "x86_64-linux";
  pkgs = import flake.inputs.nixpkgs {
    inherit system;
    config.allowUnfree = true; # Evaluate the real host NVIDIA package only.
    overlays = [ (_final: prev: {
      astrumweaverTestPackages.runtime = prev.ollama;
    }) ];
  };
  # systemd command options permit a string or a list of command strings.
  commandText = value: lib.concatStringsSep "\n" (lib.toList value);
  cases = builtins.fromJSON (builtins.readFile manifest);
  evaluate = case:
    let
      host = lib.nixosSystem {
        inherit system;
        modules = [
          flake.nixosModules.default
          (builtins.toPath case.module)
          {
            nixpkgs.pkgs = pkgs;
            system.stateVersion = "26.05";
            boot.isContainer = true;
            fileSystems."/" = { device = "none"; fsType = "tmpfs"; };
          }
        ];
      };
      c = host.config;
      control = c.services.astrumweaver.control;
      worker = c.services.astrumweaver.worker;
      cs = c.systemd.services.astrumweaver-control.serviceConfig;
      ws = c.systemd.services.astrumweaver-worker.serviceConfig;
      runtime = worker.runtime;
      packages = map toString runtime.packages;
      actual = {
        control = if !control.enable then null else {
          inherit (control) clientAuth settings migrateOnStart;
          environmentFile = cs.EnvironmentFile;
        };
        worker = if !worker.enable then null else {
          inherit (worker) workerId workerClass controlUrl capabilities gpuUuids
            totalVramMb maxSingleGpuVramMb executorFactory;
          environmentFile = ws.EnvironmentFile;
          runtime = if !runtime.enable then null else {
            inherit (runtime) provider modelRef modelFormat modelTopology
              estimatedModelSizeMb residencyPolicy gpuTopology minGpuCount
              minTotalVramMb minSingleGpuVramMb minHostRamMb preferredHostRamMb
              providerConfig modelMetadata demandMetadata;
          };
        };
      };
      failed = builtins.filter (entry: !entry.assertion) c.assertions;
      result = {
        inherit (case) name;
        inherit actual;
        # Force services and paths as well as the public options, so an
        # unevaluated packages thunk cannot hide the original unbound pkgs.
        serviceContracts =
          (!control.enable || (
            lib.hasInfix "astrumweaver-control" (commandText cs.ExecStart)
            && lib.hasInfix "--config" (commandText cs.ExecStart)
            && lib.hasInfix "astrumweaver-migrate" (commandText cs.ExecStartPre)
          ))
          && (!worker.enable || (
            lib.hasInfix "astrumweaver-worker" (commandText ws.ExecStart)
            && lib.hasInfix "--config" (commandText ws.ExecStart)
            && (worker.gpuUuids == [ ] || lib.hasInfix "gpu-preflight" (commandText ws.ExecStartPre))
          ));
        packageMatches = !worker.enable || (
          (if runtime.enable then
            packages == [ (toString (lib.getAttrFromPath case.packagePath pkgs)) ]
            && builtins.all (package:
              builtins.elem package (map toString c.systemd.services.astrumweaver-worker.path)
            ) packages
          else packages == [ ])
          && (toString worker.nvidiaSmiPackage) == (toString c.hardware.nvidia.package)
        );
      };
    in
      assert lib.assertMsg (failed == [ ])
        ("generated NixOS assertions failed in " + case.name + ": "
          + lib.concatMapStringsSep "; " (entry: entry.message) failed);
      builtins.deepSeq result result;
in
map evaluate cases
