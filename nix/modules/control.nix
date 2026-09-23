{ config, lib, pkgs, ... }:

let
  cfg = config.services.astrumweaver.control;
  astrumweaverPackage = pkgs.callPackage ../package.nix { };
  integrationPackage = pkgs.callPackage ../integration-package.nix { };
  defaultPackage = pkgs.callPackage ../control-support.nix {
    astrumweaver = astrumweaverPackage;
    integration = integrationPackage;
  };
  toml = pkgs.formats.toml { };
  generatedConfig = toml.generate "astrumweaver-control.toml" cfg.settings;
  effectiveCommand =
    if cfg.command == null
    then "${cfg.package}/bin/astrumweaver-control"
    else cfg.command;
  execStart = lib.escapeShellArgs ([ effectiveCommand "--config" generatedConfig ] ++ cfg.extraArgs);
in
{
  options.services.astrumweaver.control = {
    enable = lib.mkEnableOption "AstrumWeaver Control Plane service";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "AstrumWeaver control package from this module";
      description = "Control Plane runtime package.";
    };

    command = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/custom/bin/astrumweaver-control";
      description = "Optional daemon executable override. Defaults to the packaged astrumweaver-control entrypoint.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Non-secret TOML settings. Secrets must not be placed here because Nix store paths are world-readable.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/run/secrets/astrumweaver-control.env";
      description = "Optional protected EnvironmentFile containing database URL and authority tokens.";
    };

    migrateOnStart = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Apply packaged PostgreSQL migrations before starting Control. Disabled by default because schema mutation is an explicit authority.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
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
    users.groups.${cfg.group} = { };
    users.users.${cfg.user} = {
      isSystemUser = lib.mkDefault true;
      group = lib.mkDefault cfg.group;
      home = lib.mkDefault "/var/lib/astrumweaver";
      createHome = lib.mkDefault true;
    };

    environment.systemPackages = [ cfg.package ];

    systemd.services.astrumweaver-control = {
      description = "AstrumWeaver Control Plane";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ] ++ cfg.extraPackages;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = execStart;
        Restart = "on-failure";
        RestartSec = "5s";
        RuntimeDirectory = "astrumweaver-control";
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
      // lib.optionalAttrs cfg.migrateOnStart {
        ExecStartPre = "${cfg.package}/bin/astrumweaver-migrate";
      }
      // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
