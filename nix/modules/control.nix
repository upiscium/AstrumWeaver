{ config, lib, pkgs, ... }:

let
  cfg = config.services.astrumweaver.control;
  toml = pkgs.formats.toml { };
  generatedConfig = toml.generate "astrumweaver-control.toml" cfg.settings;
  execStart = lib.escapeShellArgs ([ cfg.command "--config" generatedConfig ] ++ cfg.extraArgs);
in
{
  options.services.astrumweaver.control = {
    enable = lib.mkEnableOption "AstrumWeaver Control Plane service";

    package = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      description = "Optional runtime support package added to the service PATH.";
    };

    command = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "/run/current-system/sw/bin/astrumweaver-control";
      description = "Absolute Control Plane daemon executable path. Required while the daemon package is not yet part of v0.1.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Non-secret TOML settings. Secrets must not be placed here because Nix store paths are world-readable.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional protected EnvironmentFile containing secrets or deployment-only overrides.";
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
    assertions = [
      {
        assertion = cfg.command != "";
        message = "services.astrumweaver.control.command must be set until a packaged Control daemon exists.";
      }
    ];

    users.groups.${cfg.group} = { };
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = "/var/lib/astrumweaver";
      createHome = true;
    };

    environment.systemPackages = lib.optional (cfg.package != null) cfg.package;

    systemd.services.astrumweaver-control = {
      description = "AstrumWeaver Control Plane";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = lib.optional (cfg.package != null) cfg.package ++ cfg.extraPackages;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = execStart;
        EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
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
      };
    };
  };
}
