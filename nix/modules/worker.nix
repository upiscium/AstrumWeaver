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
    runtimeInputs = [ pkgs.coreutils pkgs.gawk ] ++ lib.optional (cfg.nvidiaSmiPackage != null) cfg.nvidiaSmiPackage;
    text = builtins.readFile ../../libexec/gpu-preflight;
  };
  execStart = lib.escapeShellArgs ([ cfg.command "--config" generatedConfig ] ++ cfg.extraArgs);
in
{
  options.services.astrumweaver.worker = {
    enable = lib.mkEnableOption "AstrumWeaver GPU Worker service";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      description = "AstrumWeaver Worker runtime package.";
    };

    command = lib.mkOption {
      type = lib.types.str;
      default = "${cfg.package}/bin/astrumweaver-worker";
      defaultText = lib.literalExpression ''"${config.services.astrumweaver.worker.package}/bin/astrumweaver-worker"'';
      description = "Absolute Worker daemon executable path.";
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
      description = "Exact guest-visible NVIDIA GPU UUID set owned by this Worker.";
    };

    nvidiaSmiPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      example = lib.literalExpression "config.hardware.nvidia.package";
      description = "Package providing nvidia-smi for exact GPU identity preflight. Set this explicitly on NixOS NVIDIA workers.";
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
        assertion = cfg.gpuUuids != [ ];
        message = "services.astrumweaver.worker.gpuUuids must contain at least one expected GPU UUID.";
      }
      {
        assertion = builtins.length cfg.gpuUuids == builtins.length (lib.unique cfg.gpuUuids);
        message = "services.astrumweaver.worker.gpuUuids must not contain duplicates.";
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
      description = "AstrumWeaver GPU Worker";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ preflight ]
        ++ [ cfg.package ]
        ++ lib.optional (cfg.nvidiaSmiPackage != null) cfg.nvidiaSmiPackage
        ++ cfg.extraPackages;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStartPre = "+${preflight}/bin/astrumweaver-gpu-preflight ${expectedGpuUuids}";
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
      } // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
