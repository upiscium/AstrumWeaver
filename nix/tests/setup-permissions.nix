{ pkgs, integration }:

let
  fakeNvidia = pkgs.writeShellScriptBin "nvidia-smi" ''
    case "$*" in
      *"--query-gpu=uuid,minor_number"*)
        printf '%s, 0\n' GPU-permission-test
        ;;
      *"--query-gpu=uuid"*)
        printf '%s\n' GPU-permission-test
        ;;
    esac
    exit 0
  '';

  fakeDaemon = pkgs.writeShellScript "astrumweaver-permission-test-daemon" ''
    exit 0
  '';

  fakeDaemonPath = "/etc/astrumweaver-permission-test-daemon";

  node = { ... }: {
    system.stateVersion = "26.05";
    virtualisation.memorySize = 1024;

    environment.systemPackages = [
      integration
      fakeNvidia
      pkgs.bash
      pkgs.coreutils
      pkgs.gawk
      pkgs.gnugrep
      pkgs.gnused
      pkgs.shadow
      pkgs.util-linux
    ];

    environment.etc."astrumweaver-permission-test-daemon".source = fakeDaemon;
  };
in
pkgs.testers.runNixOSTest {
  name = "astrumweaver-setup-permissions";

  nodes = {
    controlFirst = node;
    workerFirst = node;
    customIdentity = node;
  };

  testScript = ''
    start_all()

    protected = (
        "/etc/astrumweaver/control.toml "
        "/etc/astrumweaver/control.env "
        "/etc/astrumweaver/worker.toml "
        "/etc/astrumweaver/worker.env "
        "/etc/astrumweaver/runtime-deployment.json "
        "/etc/astrumweaver/gpu-uuids "
        "/etc/astrumweaver/gpu-device-map"
    )

    relevant = protected + (
        " /etc/systemd/system/astrumweaver-control.service"
        " /etc/systemd/system/astrumweaver-worker.service"
        " /etc/systemd/system/astrumweaver-worker-gpu-isolation-preflight.service"
        " /etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
        " /usr/local/libexec/astrumweaver/gpu-preflight"
        " /usr/local/libexec/astrumweaver/gpu-device-map"
    )

    def prepare(machine):
        machine.wait_for_unit("multi-user.target")
        # NixOS normally links this directory into the read-only store. Give
        # the guest a writable copy to model generic systemd installation;
        # the production scripts and live ownership path remain unchanged.
        machine.succeed(
            "mkdir /run/generic-systemd && "
            "cp -a /etc/systemd/system/. /run/generic-systemd/ && "
            "mount --bind /run/generic-systemd /etc/systemd/system"
        )
        machine.succeed(
            "${pkgs.coreutils}/bin/rm -f /dev/nvidia0 && "
            "${pkgs.coreutils}/bin/mknod -m 0660 /dev/nvidia0 c 195 0 && "
            "test -c /dev/nvidia0"
        )
        machine.succeed(
            "cat >/tmp/control.toml <<'EOF'\n"
            "[control]\n"
            "listen = \"127.0.0.1:9000\"\n"
            "EOF\n"
            "cat >/tmp/control.env <<'EOF'\n"
            "ASTRUMWEAVER_TEST=control\n"
            "EOF\n"
            "cat >/tmp/worker.toml <<'EOF'\n"
            "[worker]\n"
            "id = \"worker-permission-test\"\n"
            "class = \"gpu-single\"\n"
            "gpu_uuids = [\"GPU-permission-test\"]\n"
            "EOF\n"
            "cat >/tmp/worker.env <<'EOF'\n"
            "ASTRUMWEAVER_TEST=worker\n"
            "EOF\n"
            "cat >/tmp/runtime-deployment.json <<'EOF'\n"
            "{\"schema_version\":\"v1\",\"provider_id\":\"test\"}\n"
            "EOF\n"
        )

    def setup_control(machine, user):
        machine.succeed(
            "${integration}/bin/astrumweaver-setup-control-plane "
            "--config /tmp/control.toml "
            "--environment-file /tmp/control.env "
            "--executable ${fakeDaemonPath} "
            "--root / "
            "--user " + user
        )

    def setup_worker(machine, user):
        machine.succeed(
            "${integration}/bin/astrumweaver-setup-gpu-worker "
            "--config /tmp/worker.toml "
            "--environment-file /tmp/worker.env "
            "--runtime-manifest /tmp/runtime-deployment.json "
            "--executable ${fakeDaemonPath} "
            "--root / "
            "--user " + user + " "
            "--gpu-isolation on "
            "--gpu-device GPU-permission-test=/dev/nvidia0"
        )

    def assert_permissions(machine, user):
        expected_etc = "root:" + user + " 750"
        expected_state = user + ":" + user + " 750"
        expected_file = "root:" + user + " 640"
        machine.succeed(
            "test \"$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /etc/astrumweaver)\" = '"
            + expected_etc
            + "' && test \"$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /var/lib/astrumweaver)\" = '"
            + expected_state
            + "'"
        )
        machine.succeed(
            "${pkgs.bash}/bin/bash -ceu '"
            "for path in " + protected + "; do "
            "  test \"$(${pkgs.coreutils}/bin/stat -c \"%U:%G %a\" \"$path\")\" = \""
            + expected_file
            + "\"; "
            "done'"
        )
        machine.succeed(
            "${pkgs.util-linux}/bin/runuser -u " + user + " -- "
            "${pkgs.bash}/bin/bash -ceu '"
            "test -r /etc/astrumweaver; "
            "test -r /var/lib/astrumweaver; "
            "for path in " + relevant + "; do "
            "  test -r \"$path\"; "
            "  ${pkgs.coreutils}/bin/cat \"$path\" >/dev/null; "
            "done'"
        )
        machine.succeed(
            "${pkgs.bash}/bin/bash -ceu '"
            "for path in " + protected + "; do "
            "  if ${pkgs.util-linux}/bin/runuser -u nobody -- ${pkgs.coreutils}/bin/test -r \"$path\"; then exit 1; fi; "
            "done; "
            "if ${pkgs.util-linux}/bin/runuser -u nobody -- ${pkgs.coreutils}/bin/test -r /var/lib/astrumweaver; then exit 1; fi'"
        )

    def assert_units(machine, user):
        machine.succeed(
            "${pkgs.gnugrep}/bin/grep -F -- 'User=" + user + "' "
            "/etc/systemd/system/astrumweaver-control.service && "
            "${pkgs.gnugrep}/bin/grep -F -- 'Group=" + user + "' "
            "/etc/systemd/system/astrumweaver-control.service && "
            "${pkgs.gnugrep}/bin/grep -F -- 'User=" + user + "' "
            "/etc/systemd/system/astrumweaver-worker.service && "
            "${pkgs.gnugrep}/bin/grep -F -- 'Group=" + user + "' "
            "/etc/systemd/system/astrumweaver-worker.service"
        )
        machine.succeed(
            "${pkgs.gnugrep}/bin/grep -Fx -- 'GPU-permission-test=/dev/nvidia0' "
            "/etc/astrumweaver/gpu-device-map && "
            "${pkgs.gnugrep}/bin/grep -F -- '${fakeDaemonPath} --config /etc/astrumweaver/worker.toml --runtime-manifest /etc/astrumweaver/runtime-deployment.json' "
            "/etc/systemd/system/astrumweaver-worker.service && "
            "${pkgs.gnugrep}/bin/grep -F -- 'DevicePolicy=closed' "
            "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf && "
            "${pkgs.gnugrep}/bin/grep -F -- 'DeviceAllow=/dev/nvidia0 rw' "
            "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf && "
            "${pkgs.gnugrep}/bin/grep -F -- 'Environment=CUDA_VISIBLE_DEVICES=GPU-permission-test' "
            "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
        )
        machine.succeed(
            "systemctl daemon-reload && "
            "systemctl cat astrumweaver-control.service >/dev/null && "
            "systemctl cat astrumweaver-worker.service >/dev/null && "
            "systemctl cat astrumweaver-worker-gpu-isolation-preflight.service >/dev/null"
        )

    def repair_directories(machine, user):
        machine.succeed(
            "${pkgs.coreutils}/bin/chown root:root /etc/astrumweaver /var/lib/astrumweaver && "
            "${pkgs.coreutils}/bin/chmod 0750 /etc/astrumweaver /var/lib/astrumweaver"
        )
        setup_worker(machine, user)
        assert_permissions(machine, user)

    def exercise_order(machine, first):
        prepare(machine)
        if first == "control":
            setup_control(machine, "astrumweaver")
            machine.succeed("runuser -u astrumweaver -- cat /etc/astrumweaver/control.toml /etc/astrumweaver/control.env >/dev/null")
            setup_worker(machine, "astrumweaver")
        else:
            setup_worker(machine, "astrumweaver")
            machine.succeed("runuser -u astrumweaver -- cat /etc/astrumweaver/worker.toml /etc/astrumweaver/worker.env /etc/astrumweaver/gpu-uuids /etc/astrumweaver/gpu-device-map /etc/astrumweaver/runtime-deployment.json >/dev/null")
            setup_control(machine, "astrumweaver")
        assert_permissions(machine, "astrumweaver")
        assert_units(machine, "astrumweaver")
        repair_directories(machine, "astrumweaver")

    exercise_order(controlFirst, "control")
    exercise_order(workerFirst, "worker")

    prepare(customIdentity)
    setup_control(customIdentity, "shared-service")
    customIdentity.succeed(
        "before=$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /etc/astrumweaver); "
        "before_state=$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /var/lib/astrumweaver); "
        "test \"$before\" = 'root:shared-service 750'; "
        "test \"$before_state\" = 'shared-service:shared-service 750'; "
        "if ${integration}/bin/astrumweaver-setup-gpu-worker "
        "--config /tmp/worker.toml --environment-file /tmp/worker.env "
        "--runtime-manifest /tmp/runtime-deployment.json "
        "--executable ${fakeDaemonPath} --root / --user worker-service "
        "--gpu-isolation on --gpu-device GPU-permission-test=/dev/nvidia0 "
        ">/tmp/conflict.log 2>&1; then exit 1; fi; "
        "${pkgs.gnugrep}/bin/grep -F -- 'conflicting existing role unit identity' /tmp/conflict.log; "
        "test \"$before\" = \"$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /etc/astrumweaver)\"; "
        "test \"$before_state\" = \"$(${pkgs.coreutils}/bin/stat -c '%U:%G %a' /var/lib/astrumweaver)\"; "
        "test ! -e /etc/astrumweaver/worker.toml; "
        "test ! -e /etc/astrumweaver/worker.env; "
        "test ! -e /etc/astrumweaver/runtime-deployment.json; "
        "test ! -e /etc/astrumweaver/gpu-uuids; "
        "test ! -e /etc/astrumweaver/gpu-device-map; "
        "test ! -e /etc/systemd/system/astrumweaver-worker.service"
    )
    setup_worker(customIdentity, "shared-service")
    assert_permissions(customIdentity, "shared-service")
    assert_units(customIdentity, "shared-service")
  '';
}
