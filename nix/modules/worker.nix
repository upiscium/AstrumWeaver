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
  effectiveRuntimeGpuTopology =
    if cfg.runtime.gpuTopology != null
    then cfg.runtime.gpuTopology
    else if cfg.gpuUuids == [ ]
    then "none"
    else if builtins.length cfg.gpuUuids == 1
    then "single_gpu"
    else "multi_gpu";
  effectiveRuntimeMinGpuCount =
    if effectiveRuntimeGpuTopology == "multi_gpu"
    then builtins.max 2 cfg.runtime.minGpuCount
    else cfg.runtime.minGpuCount;
  runtimeManifestData = {
    schema_version = "v1";
    provider_id = cfg.runtime.provider;
    provider_config = cfg.runtime.providerConfig;
    demand = {
      model = {
        model_ref = cfg.runtime.modelRef;
        model_format = cfg.runtime.modelFormat;
        topology = cfg.runtime.modelTopology;
        estimated_size_mb = cfg.runtime.estimatedModelSizeMb;
        metadata = cfg.runtime.modelMetadata;
      };
      residency_policy = cfg.runtime.residencyPolicy;
      gpu_topology = effectiveRuntimeGpuTopology;
      min_gpu_count = effectiveRuntimeMinGpuCount;
      min_total_vram_mb = cfg.runtime.minTotalVramMb;
      min_single_gpu_vram_mb = cfg.runtime.minSingleGpuVramMb;
      min_host_ram_mb = cfg.runtime.minHostRamMb;
      preferred_host_ram_mb = cfg.runtime.preferredHostRamMb;
      metadata = cfg.runtime.demandMetadata;
    };
    setup_intent = null;
  };
  runtimeManifest = pkgs.writeText "astrumweaver-runtime-deployment.json" (
    builtins.toJSON runtimeManifestData
  );
  coreSettings = {
    worker = {
      id = cfg.workerId;
      class = cfg.workerClass;
      control_url = cfg.controlUrl;
      gpu_uuids = cfg.gpuUuids;
      gpu_count = builtins.length cfg.gpuUuids;
      total_vram_mb = cfg.totalVramMb;
      max_single_gpu_vram_mb = cfg.maxSingleGpuVramMb;
      capabilities = cfg.capabilities;
      labels = cfg.labels;
      accelerators = cfg.accelerators;
      max_concurrency = cfg.maxConcurrency;
      gpu_preflight = cfg.gpuUuids != [ ];
      health_host = cfg.healthHost;
      health_port = cfg.healthPort;
    };
    executor = {
      factory = cfg.executorFactory;
      settings = cfg.executorSettings;
    };
  } // lib.optionalAttrs cfg.runtime.enable {
    runtime = {
      manifest = runtimeManifest;
      startup_timeout_seconds = cfg.runtime.startupTimeoutSeconds;
      shutdown_timeout_seconds = cfg.runtime.shutdownTimeoutSeconds;
    };
  };
  generatedConfig = toml.generate "astrumweaver-worker.toml" (
    lib.recursiveUpdate cfg.settings coreSettings
  );
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
  gpuUuidArgs = lib.concatMapStringsSep " " (uuid:
    "--gpu-uuid ${lib.escapeShellArg uuid}"
  ) cfg.gpuUuids;
  nvidiaSmiCommand =
    if cfg.nvidiaSmiPackage == null
    then "nvidia-smi"
    else "${cfg.nvidiaSmiPackage}/bin/nvidia-smi";
  borrowableMode = pkgs.writeShellApplication {
    name = "astrumweaver-gpu-mode";
    text = ''
      if [ "$#" -ne 1 ]; then
        echo "usage: astrumweaver-gpu-mode {development|astrumweaver|status}" >&2
        exit 2
      fi
      exec ${cfg.package}/bin/astrumweaver-worker-mode "$1" \
        --service astrumweaver-worker.service \
        ${gpuUuidArgs} \
        --health-url ${lib.escapeShellArg "http://${cfg.healthHost}:${toString cfg.healthPort}"} \
        --systemctl ${pkgs.systemd}/bin/systemctl \
        --nvidia-smi ${nvidiaSmiCommand} \
        --drain-timeout-seconds ${toString cfg.borrowable.drainTimeoutSeconds} \
        --start-timeout-seconds ${toString cfg.borrowable.startTimeoutSeconds}
    '';
  };
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

    workerId = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "worker-1";
      description = "Stable scheduler-visible Worker identity.";
    };

    workerClass = lib.mkOption {
      type = lib.types.str;
      default = "generic";
      example = "modern-single";
      description = "Hardware/runtime topology class; not an application role.";
    };

    controlUrl = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "https://control.example.invalid";
      description = "AstrumWeaver Control API base URL.";
    };

    capabilities = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "llm.chat" "image.generate" ];
      description = "Capabilities advertised by this Worker.";
    };

    labels = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = { runtime_family = "modern"; };
    };

    gpuUuids = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "GPU-example-a" "GPU-example-b" ];
      description = "Exact guest-visible NVIDIA GPU UUID set. Leave empty for a non-GPU Worker.";
    };

    accelerators = lib.mkOption {
      type = lib.types.listOf (lib.types.submodule {
        options = {
          uuid = lib.mkOption {
            type = lib.types.str;
            description = "Exact GPU UUID for this accelerator fact.";
          };
          memory_mb = lib.mkOption {
            type = lib.types.ints.positive;
            description = "Per-device VRAM in MiB.";
          };
          compute_capability = lib.mkOption {
            type = lib.types.nullOr lib.types.str;
            default = null;
            description = "Validated NVIDIA compute capability, e.g. 8.6.";
          };
          device_class = lib.mkOption {
            type = lib.types.nullOr lib.types.str;
            default = null;
            description = "Auditable accelerator device class/model when known.";
          };
        };
      });
      default = [ ];
      description = "Optional ordered per-device accelerator facts. UUID order must match gpuUuids.";
    };

    totalVramMb = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 0;
    };

    maxSingleGpuVramMb = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 0;
    };

    maxConcurrency = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1;
      description = "Maximum concurrent jobs. The v1 daemon currently supports 1.";
    };

    executorFactory = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "my_executor:create_executor";
      description = "Configured JobExecutor factory in module:attribute form.";
    };

    executorSettings = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Executor-specific non-secret configuration.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Additional non-secret Worker TOML settings. Core module-owned fields override conflicting values.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/run/secrets/astrumweaver-worker.env";
      description = "Protected EnvironmentFile containing ASTRUMWEAVER_WORKER_TOKEN and other secret overrides.";
    };

    nvidiaSmiPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      example = lib.literalExpression "config.hardware.nvidia.package";
      description = "Package providing nvidia-smi. Required when gpuUuids is non-empty.";
    };

    supplementaryGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "video" "render" ];
      description = "Host-specific groups needed for accelerator device access.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "astrumweaver";
    };

    healthHost = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
    };

    healthPort = lib.mkOption {
      type = lib.types.port;
      default = 9100;
    };

    runtime = {
      enable = lib.mkEnableOption "RuntimeProvider-managed Worker execution";

      provider = lib.mkOption {
        type = lib.types.enum [
          "ollama"
          "llama-cpp"
          "vllm"
          "freetoken"
          "exllamav3"
        ];
        default = "ollama";
        description = "Explicit first-class RuntimeProvider selected by the operator.";
      };

      packages = lib.mkOption {
        type = lib.types.listOf lib.types.package;
        default = [ ];
        description = "Packages that provide the selected runtime executable/backend. They are added only to the Worker service closure.";
      };

      providerConfig = lib.mkOption {
        type = lib.types.attrs;
        default = { };
        description = "Provider-specific non-secret configuration persisted in the runtime deployment manifest.";
      };

      modelRef = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "Explicit model reference used by the RuntimeProvider.";
      };

      modelFormat = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "Provider-neutral model/package format.";
      };

      modelTopology = lib.mkOption {
        type = lib.types.enum [ "dense" "moe" ];
        default = "dense";
      };

      estimatedModelSizeMb = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.positive;
        default = null;
      };

      residencyPolicy = lib.mkOption {
        type = lib.types.enum [
          "vram_only"
          "prefer_vram"
          "cpu_gpu_hybrid"
        ];
        default = "prefer_vram";
      };

      gpuTopology = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum [
          "none"
          "single_gpu"
          "multi_gpu"
        ]);
        default = null;
        description = "GPU topology demand. null derives it from the Worker's exact gpuUuids.";
      };

      minGpuCount = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
      };

      minTotalVramMb = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
      };

      minSingleGpuVramMb = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
      };

      minHostRamMb = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
      };

      preferredHostRamMb = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
      };

      modelMetadata = lib.mkOption {
        type = lib.types.attrs;
        default = { };
      };

      demandMetadata = lib.mkOption {
        type = lib.types.attrs;
        default = { };
      };

      startupTimeoutSeconds = lib.mkOption {
        type = lib.types.ints.positive;
        default = 600;
      };

      shutdownTimeoutSeconds = lib.mkOption {
        type = lib.types.ints.positive;
        default = 60;
      };
    };

    borrowable = {
      enable = lib.mkEnableOption "borrowable GPU ownership mode for a development node";

      drainTimeoutSeconds = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
        description = "Seconds to wait for the current job to drain. 0 waits indefinitely and never forces the job.";
      };

      startTimeoutSeconds = lib.mkOption {
        type = lib.types.ints.positive;
        default = 60;
        description = "Seconds to wait for the Worker service and local readiness when returning GPU ownership to AstrumWeaver.";
      };
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
        assertion = cfg.workerId != "";
        message = "services.astrumweaver.worker.workerId must be set.";
      }
      {
        assertion = cfg.controlUrl != "";
        message = "services.astrumweaver.worker.controlUrl must be set.";
      }
      {
        assertion = cfg.capabilities != [ ];
        message = "services.astrumweaver.worker.capabilities must not be empty.";
      }
      {
        assertion =
          if cfg.runtime.enable
          then cfg.executorFactory == ""
          else cfg.executorFactory != "";
        message = "Configure exactly one Worker execution path: runtime.enable=true with no executorFactory, or a non-empty executorFactory.";
      }
      {
        assertion = !cfg.runtime.enable || cfg.runtime.modelRef != "";
        message = "services.astrumweaver.worker.runtime.modelRef must be set when RuntimeProvider execution is enabled.";
      }
      {
        assertion = !cfg.runtime.enable || cfg.runtime.modelFormat != "";
        message = "services.astrumweaver.worker.runtime.modelFormat must be set when RuntimeProvider execution is enabled.";
      }
      {
        assertion = !cfg.runtime.enable || cfg.runtime.packages != [ ];
        message = "services.astrumweaver.worker.runtime.packages must provide the selected runtime closure.";
      }
      {
        assertion = !cfg.runtime.enable || cfg.runtime.preferredHostRamMb >= cfg.runtime.minHostRamMb;
        message = "runtime.preferredHostRamMb must be >= runtime.minHostRamMb.";
      }
      {
        assertion =
          cfg.accelerators == [ ]
          || (
            builtins.length cfg.accelerators == builtins.length cfg.gpuUuids
            && map (item: item.uuid) cfg.accelerators == cfg.gpuUuids
          );
        message = "services.astrumweaver.worker.accelerators must be empty or exactly match gpuUuids order.";
      }
      {
        assertion = cfg.maxConcurrency == 1;
        message = "AstrumWeaver v1 Worker daemon currently requires maxConcurrency = 1.";
      }
      {
        assertion = builtins.length cfg.gpuUuids == builtins.length (lib.unique cfg.gpuUuids);
        message = "services.astrumweaver.worker.gpuUuids must not contain duplicates.";
      }
      {
        assertion = cfg.gpuUuids == [ ] || cfg.nvidiaSmiPackage != null;
        message = "services.astrumweaver.worker.nvidiaSmiPackage is required for GPU Workers.";
      }
      {
        assertion = !cfg.borrowable.enable || cfg.gpuUuids != [ ];
        message = "services.astrumweaver.worker.borrowable requires at least one GPU UUID.";
      }
      {
        assertion = !cfg.borrowable.enable || cfg.nvidiaSmiPackage != null;
        message = "services.astrumweaver.worker.borrowable requires nvidiaSmiPackage.";
      }
      {
        assertion = !cfg.borrowable.enable || builtins.elem cfg.healthHost [ "127.0.0.1" "localhost" ];
        message = "borrowable Worker healthHost must remain local (127.0.0.1 or localhost).";
      }
      {
        assertion =
          if cfg.gpuUuids == [ ]
          then cfg.totalVramMb == 0 && cfg.maxSingleGpuVramMb == 0
          else cfg.totalVramMb > 0
            && cfg.maxSingleGpuVramMb > 0
            && cfg.maxSingleGpuVramMb <= cfg.totalVramMb;
        message = "Worker VRAM shape must be zero for non-GPU Workers and positive/consistent for GPU Workers.";
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

    environment.systemPackages =
      [ cfg.package ]
      ++ lib.optional cfg.borrowable.enable borrowableMode;

    systemd.services.astrumweaver-worker = {
      description = "AstrumWeaver Worker";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ]
        ++ lib.optional (cfg.nvidiaSmiPackage != null) cfg.nvidiaSmiPackage
        ++ cfg.runtime.packages
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
