{ config, lib, pkgs, ... }:

let
  cfg = config.services.astrumweaver.worker;
  astrumweaverPackage = pkgs.callPackage ../package.nix { };
  integrationPackage = pkgs.callPackage ../integration-package.nix { };
  defaultPackage = pkgs.callPackage ../worker-support.nix {
    astrumweaver = astrumweaverPackage;
    integration = integrationPackage;
  };
  toml = pkgs.formats.toml { };
  generatedConfig = toml.generate "astrumweaver-worker.toml" cfg.settings;
  expectedGpuUuids = pkgs.writeText "astrumweaver-gpu-uuids" (
    lib.concatStringsSep "\n" (lib.sort builtins.lessThan cfg.gpuUuids) + "\n"
  );
  preflight = pkgs.writeShellApplication {
    name = "astrumweaver-gpu-preflight";
    runtimeInputs = [ pkgs.coreutils pkgs.gawk ]
      ++ lib.optional (cfg.nvidiaSmiPackage != null) cfg.nvidiaSmiPackage;
    text = builtins.readFile ../../libexec/gpu-preflight;
  };
  effectiveCommand =
    if cfg.command == null
    then "${cfg.package}/bin/astrumweaver-worker"
    else cfg.command;
  execStart = lib.escapeShellArgs ([ effectiveCommand "--config" generatedConfig ] ++ cfg.extraArgs);
in
{
  options.services.astrumweaver.worker = {
    enable = lib.mkEnableOption "AstrumWeaver Worker service";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "AstrumWeaver worker package from this module";
      description = "Worker runtime package.";
    };

    command = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/custom/bin/astrumweaver-worker";
      description = "Optional daemon executable override. Defaults to the packaged astrumweaver-worker entrypoint.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Non-secret Worker TOML settings. Secrets must not be stored in the Nix store.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
    };

    gpuUuids = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "GPU-example-a" "GPU-example-b" ];
      description = "Exact guest-visible NVIDIA GPU UUID set. Leave empty for a non-GPU Worker.";
    };

    nvidiaSmiPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      example = lib.literalExpression "config.hardware.nvidia.package";
      description = "Package providing nvidia-smi. Required when gpuUuids is non-empty.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
    };

    supplementaryGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "video" "render" ];
      description = "Host-specific groups needed for accelerator device access.";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
    };

    extraPackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ ];
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = builtins.length cfg.gpuUuids == builtins.length (lib.unique cfg.gpuUuids);
        message = "services.astrumweaver.worker.gpuUuids must not contain duplicates.";
      }
      {
        assertion = cfg.gpuUuids == [ ] || cfg.nvidiaSmiPackage != null;
        message = "services.astrumweaver.worker.nvidiaSmiPackage is required for GPU Workers.";
      }
    ];

    users.groups.${cfg.group} = { };
    users.users.${cfg.user} = {
      isSystemUser = lib.mkDefault true;
      group = lib.mkDefault cfg.group;
      extraGroups = cfg.supplementaryGroups;
      home = lib.mkDefault "/var/lib/astrumweaver";
      createHome = lib.mkDefault true;
    };

    environment.systemPackages = [ cfg.package ];

    systemd.services.astrumweaver-worker = {
      description = "AstrumWeaver Worker";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ]
        ++ lib.optional (cfg.nvidiaSmiPackage != null) cfg.nvidiaSmiPackage
        ++ cfg.extraPackages;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = execStart;
        Restart = "on-failure";
        RestartSec = "5s";
        RuntimeDirectory = "astrumweaver-worker";
        RuntimeDirectoryMode = "0750";
        StateDirectory = "astrumweaver";
        StateDirectoryMode = "0750";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
      }
      // lib.optionalAttrs (cfg.gpuUuids != [ ]) {
        ExecStartPre = "+${preflight}/bin/astrumweaver-gpu-preflight ${expectedGpuUuids}";
      }
      // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
