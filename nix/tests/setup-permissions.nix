{ pkgs, integration }:

let
  fakeNvidia = pkgs.writeShellScriptBin "nvidia-smi" ''
    case "$*" in
      *"--query-gpu=uuid,minor_number"*) exit 2 ;;
      *"--query-gpu=uuid"*) printf '%s\n' GPU-permission-test ;;
    esac
    exit 0
  '';
  fakeDaemon = pkgs.writeShellScript "astrumweaver-permission-test-daemon" ''
    exit 0
  '';
  fakeDaemonPath = "/etc/astrumweaver-permission-test-daemon";
  baseNode = { ... }: {
    system.stateVersion = "26.05";
    virtualisation.memorySize = 1024;
    users.mutableUsers = true;
    environment.systemPackages = [
      integration fakeNvidia pkgs.bash pkgs.coreutils pkgs.diffutils pkgs.gawk
      pkgs.gnugrep pkgs.gnused pkgs.shadow pkgs.systemd pkgs.util-linux
    ];
    environment.etc."astrumweaver-permission-test-daemon".source = fakeDaemon;
  };
  customIdentityNode = { ... }: {
    imports = [ baseNode ];
    # Both same-name groups exist without users: this exercises useradd's
    # --gid branch for both custom roles. The default nodes exercise --user-group.
    users.groups."custom-control" = { };
    users.groups."custom-worker" = { };
  };
  negativeFixturesNode = { ... }: {
    imports = [ baseNode ];
    users.groups."bad-primary" = { };
    users.groups."bad-primary-source" = { };
    users.users."bad-primary" = {
      isSystemUser = true; group = "bad-primary-source";
      home = "/var/empty"; createHome = false;
    };
    users.groups."cross-control" = { };
    users.groups."cross-worker" = { };
    users.users."cross-control" = {
      isSystemUser = true; group = "cross-control";
      home = "/var/empty"; createHome = false;
    };
    users.users."cross-worker" = {
      isSystemUser = true; group = "cross-worker";
      home = "/var/empty"; createHome = false;
    };
  };
in
pkgs.testers.runNixOSTest {
  name = "astrumweaver-setup-permissions";
  nodes = {
    controlFirst = baseNode;
    workerFirst = baseNode;
    customIdentity = customIdentityNode;
    negativeFixtures = negativeFixturesNode;
  };
  testScript = ''
    start_all()

    bash = "${pkgs.bash}/bin/bash"
    cat = "${pkgs.coreutils}/bin/cat"
    chmod = "${pkgs.coreutils}/bin/chmod"
    chown = "${pkgs.coreutils}/bin/chown"
    cp = "${pkgs.coreutils}/bin/cp"
    cmp = "${pkgs.diffutils}/bin/cmp"
    cut = "${pkgs.coreutils}/bin/cut"
    grep = "${pkgs.gnugrep}/bin/grep"
    id = "${pkgs.coreutils}/bin/id"
    ls = "${pkgs.coreutils}/bin/ls"
    mkdir = "${pkgs.coreutils}/bin/mkdir"
    mknod = "${pkgs.coreutils}/bin/mknod"
    rm = "${pkgs.coreutils}/bin/rm"
    runuser = "${pkgs.util-linux}/bin/runuser"
    sha = "${pkgs.coreutils}/bin/sha256sum"
    stat = "${pkgs.coreutils}/bin/stat"
    systemd_run = "${pkgs.systemd}/bin/systemd-run"
    tr = "${pkgs.coreutils}/bin/tr"

    control_bin = "${integration}/bin/astrumweaver-setup-control-plane"
    worker_bin = "${integration}/bin/astrumweaver-setup-gpu-worker"
    fake_daemon_path = "${fakeDaemonPath}"
    control_files = ["/etc/astrumweaver/control.toml", "/etc/astrumweaver/control.env"]
    worker_files = [
        "/etc/astrumweaver/worker.toml", "/etc/astrumweaver/worker.env",
        "/etc/astrumweaver/gpu-uuids", "/etc/astrumweaver/gpu-device-map",
        "/etc/astrumweaver/runtime-deployment.json",
    ]
    fake_gpu_proc = "/run/astrumweaver-fake-nvidia-gpus"
    all_files = control_files + worker_files
    control_unit = "/etc/systemd/system/astrumweaver-control.service"
    worker_unit = "/etc/systemd/system/astrumweaver-worker.service"
    isolation_unit = "/etc/systemd/system/astrumweaver-worker-gpu-isolation-preflight.service"
    isolation_dropin = "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
    dropin_override = "/etc/systemd/system/astrumweaver-worker.service.d/override.conf"
    control_state = "/var/lib/astrumweaver-control"
    worker_state = "/var/lib/astrumweaver"
    snapshot_paths = [
        "/etc/astrumweaver", control_state, worker_state, control_unit, worker_unit,
        isolation_unit, isolation_dropin, dropin_override,
    ] + all_files
    sentinel_paths = snapshot_paths + ["/etc/astrumweaver/unchanged"]

    def control_cmd(user):
        return (f"{control_bin} --config /tmp/control.toml --environment-file /tmp/control.env "
                f"--executable {fake_daemon_path} --root / --user {user}")

    def worker_cmd(user):
        return (f"ASTRUMWEAVER_GPU_PROC_ROOT={fake_gpu_proc} {worker_bin} "
                "--config /tmp/worker.toml --environment-file /tmp/worker.env "
                f"--runtime-manifest /tmp/runtime-deployment.json --executable {fake_daemon_path} "
                f"--root / --user {user} --gpu-isolation on "
                "--gpu-device GPU-permission-test=/dev/nvidia0")

    def run_as(user, command):
        return f"{runuser} -u {user} -- {bash} -ceu '{command}'"

    def deny_read(machine, user, path):
        machine.succeed(f"if {runuser} -u {user} -- {cat} {path} >/dev/null 2>&1; then exit 1; fi")

    def deny_write(machine, user, path):
        command = f"printf denied >> {path}"
        machine.succeed(f"if {run_as(user, command)}; then exit 1; fi")

    def prepare(machine):
        machine.wait_for_unit("multi-user.target")
        # NixOS links this directory into the store; generic systemd installs
        # need a writable copy for the setup scripts' existing-file checks.
        machine.succeed(
            f"{mkdir} /run/generic-systemd && {cp} -a "
            "/etc/systemd/system/. /run/generic-systemd/ && "
            "mount --bind /run/generic-systemd /etc/systemd/system"
        )
        machine.succeed(
            f"{rm} -f /dev/nvidia0 && "
            f"{mknod} -m 0660 /dev/nvidia0 c 195 0 && test -c /dev/nvidia0"
        )
        machine.succeed(
            f"{mkdir} -p {fake_gpu_proc}/gpu0 && "
            f"{cat} >{fake_gpu_proc}/gpu0/information <<'EOF'\n"
            "GPU UUID : GPU-permission-test\nDevice Minor : 0\nEOF\n"
        )
        machine.succeed(
            f"{cat} >/tmp/control.toml <<'EOF'\n[control]\nlisten = \"127.0.0.1:9000\"\nEOF\n"
            f"{cat} >/tmp/control.env <<'EOF'\nASTRUMWEAVER_TEST=control\nEOF\n"
            f"{cat} >/tmp/worker.toml <<'EOF'\n[worker]\nid = \"worker-permission-test\"\n"
            "class = \"gpu-single\"\ngpu_uuids = [\"GPU-permission-test\"]\nEOF\n"
            f"{cat} >/tmp/worker.env <<'EOF'\nASTRUMWEAVER_TEST=worker\nEOF\n"
            f"{cat} >/tmp/runtime-deployment.json <<'EOF'\n"
            "{\"schema_version\":\"v1\",\"provider_id\":\"test\"}\nEOF\n"
        )

    def install(machine, role, user):
        machine.succeed(control_cmd(user) if role == "control" else worker_cmd(user))

    def stat_is(machine, path, value):
        machine.succeed(f"test \"$({stat} -c '%U:%G %a' {path})\" = '{value}'")

    def unit_is(machine, path, user, state):
        for key, value in [("User", user), ("Group", user),
                           ("SupplementaryGroups", "astrumweaver-config"),
                           ("StateDirectory", state)]:
            machine.succeed(f"{grep} -Fx -- '{key}={value}' {path}")

    def role_layout(machine, role, user):
        files, state, unit, state_name = (
            (control_files, control_state, control_unit, "astrumweaver-control")
            if role == "control" else
            (worker_files, worker_state, worker_unit, "astrumweaver")
        )
        stat_is(machine, "/etc/astrumweaver", "root:astrumweaver-config 710")
        stat_is(machine, state, f"{user}:{user} 750")
        for path in files:
            stat_is(machine, path, f"root:{user} 640")
        machine.succeed(f"test \"$(getent passwd {user} | {cut} -d: -f6)\" = '{state}'")
        machine.succeed(f"{id} -nG {user} | {tr} ' ' '\\n' | {grep} -Fx -- astrumweaver-config")
        unit_is(machine, unit, user, state_name)

    def assert_all(machine, control_user, worker_user):
        role_layout(machine, "control", control_user)
        role_layout(machine, "worker", worker_user)
        machine.succeed(f"{grep} -Fx -- 'GPU-permission-test=/dev/nvidia0' /etc/astrumweaver/gpu-device-map")
        machine.succeed("test -f /usr/local/libexec/astrumweaver/gpu_mapping.py")
        machine.succeed(
            f"{grep} -F -- '{fake_daemon_path} --config /etc/astrumweaver/worker.toml "
            "--runtime-manifest /etc/astrumweaver/runtime-deployment.json' "
            f"{worker_unit} && {grep} -F -- 'DevicePolicy=closed' {isolation_dropin} && "
            f"{grep} -F -- 'DeviceAllow=/dev/nvidia0 rw' {isolation_dropin} && "
            f"{grep} -F -- 'Environment=CUDA_VISIBLE_DEVICES=GPU-permission-test' {isolation_dropin}"
        )
        machine.succeed(
            "systemctl daemon-reload && systemctl cat astrumweaver-control.service >/dev/null && "
            "systemctl cat astrumweaver-worker.service >/dev/null && "
            "systemctl cat astrumweaver-worker-gpu-isolation-preflight.service >/dev/null"
        )

    def assert_only(machine, role, user):
        role_layout(machine, role, user)
        own = control_files if role == "control" else worker_files
        peer = worker_files if role == "control" else control_files
        state = worker_state if role == "control" else control_state
        unit = worker_unit if role == "control" else control_unit
        machine.succeed(run_as(user, f"test -x /etc/astrumweaver; cd /etc/astrumweaver; "
                                    f"if {ls} >/dev/null 2>&1; then exit 1; fi; "
                                    f"{cat} {' '.join(own)} >/dev/null"))
        machine.succeed(" && ".join([f"test ! -e {path}" for path in peer + [state, unit]]))

    def assert_access(machine, control_user, worker_user):
        for user, own, peer in [
            (control_user, control_files, worker_files),
            (worker_user, worker_files, control_files),
        ]:
            machine.succeed(run_as(user, f"test -x /etc/astrumweaver; cd /etc/astrumweaver; "
                                        f"if {ls} >/dev/null 2>&1; then exit 1; fi; "
                                        f"{cat} {' '.join(own)} >/dev/null"))
            for path in peer:
                deny_read(machine, user, path)
            for path in own + peer:
                deny_read(machine, "nobody", path)
        for owner, other, state in [
            (control_user, worker_user, control_state),
            (worker_user, control_user, worker_state),
        ]:
            marker = f"{state}/permission-write-check"
            machine.succeed(run_as(owner, f"printf owner > {marker}; cat {marker} >/dev/null; printf owner >> {marker}"))
            deny_read(machine, other, marker)
            deny_write(machine, other, marker)
            deny_read(machine, "nobody", marker)

    def service_cat(user, path):
        return (f"{systemd_run} --wait --pipe --uid {user} --gid {user} "
                f"-p SupplementaryGroups=astrumweaver-config {cat} {path}")

    def assert_systemd_access(machine, control_user, worker_user):
        for user, own, peer in [
            (control_user, control_files, worker_files),
            (worker_user, worker_files, control_files),
        ]:
            for path in own:
                machine.succeed(f"{service_cat(user, path)} >/dev/null")
            for path in peer:
                machine.succeed(f"if {service_cat(user, path)} >/dev/null 2>&1; then exit 1; fi")

    def damage(machine):
        machine.succeed(
            f"{chown} root:root /etc/astrumweaver {control_state} {worker_state} {' '.join(all_files)} && "
            f"{chmod} 0777 /etc/astrumweaver && {chmod} 0700 {control_state} {worker_state} && "
            f"{chmod} 0600 {' '.join(all_files)}"
        )

    def repair_both_orders(machine):
        for order in [("control", "worker"), ("worker", "control")]:
            damage(machine)
            for role in order:
                install(machine, role, "astrumweaver-control" if role == "control" else "astrumweaver")
            assert_all(machine, "astrumweaver-control", "astrumweaver")

    def exercise_order(machine, first):
        prepare(machine)
        machine.succeed("! getent passwd astrumweaver-control && ! getent group astrumweaver-control && "
                        "! getent passwd astrumweaver && ! getent group astrumweaver")
        install(machine, first, "astrumweaver-control" if first == "control" else "astrumweaver")
        assert_only(machine, first, "astrumweaver-control" if first == "control" else "astrumweaver")
        second = "worker" if first == "control" else "control"
        install(machine, second, "astrumweaver" if second == "worker" else "astrumweaver-control")
        assert_all(machine, "astrumweaver-control", "astrumweaver")
        repair_both_orders(machine)
        assert_access(machine, "astrumweaver-control", "astrumweaver")
        assert_systemd_access(machine, "astrumweaver-control", "astrumweaver")

    def clear(machine):
        machine.succeed(
            f"{rm} -rf /etc/astrumweaver {control_state} {worker_state} /usr/local/libexec/astrumweaver; "
            f"{rm} -f {control_unit} {worker_unit} {isolation_unit}; "
            f"{rm} -rf /etc/systemd/system/astrumweaver-worker.service.d"
        )

    def snapshot(machine, name, paths=snapshot_paths, users=[], groups=[]):
        records = []
        for path in paths:
            records.append(
                f"if test -e {path}; then {stat} -c 'path=%n type=%F owner=%U:%G mode=%a size=%s' {path}; "
                f"if test -f {path}; then {sha} {path}; fi; else printf 'absent %s\\n' {path}; fi"
            )
        for user in users:
            records.append(f"printf 'passwd:{user} '; getent passwd {user} 2>/dev/null || true")
            records.append(f"printf 'groups:{user} '; {id} -nG {user} 2>/dev/null || true")
        for group in groups:
            records.append(f"printf 'group:{group} '; getent group {group} 2>/dev/null || true")
        machine.succeed("{ " + "; ".join(records) + f"; }} > /tmp/{name}")

    def check_snapshot(machine, name, paths=snapshot_paths, users=[], groups=[]):
        snapshot(machine, name + ".after", paths, users, groups)
        machine.succeed(f"{cmp} -s /tmp/{name} /tmp/{name}.after")

    def reject(machine, name, command, users=[], needle=None, paths=snapshot_paths, groups=[]):
        snapshot(machine, name, paths, users, groups)
        machine.succeed(f"if {command} >/tmp/{name}.err 2>&1; then exit 1; fi")
        if needle:
            machine.succeed(f"{grep} -F -- '{needle}' /tmp/{name}.err")
        check_snapshot(machine, name, paths, users, groups)

    def reject_numeric(machine, name, role, user, kind, users, groups):
        command = control_cmd(user) if role == "control" else worker_cmd(user)
        reject(machine, name, command, users=users,
               needle=f"numeric {kind} alias", groups=groups)

    def remove_fixture_alias(machine, user=None, group=None):
        if user:
            machine.succeed(f"userdel {user}")
        if group:
            # This test-only alias intentionally shares a primary GID. Remove
            # its named entry, retaining the real role group with that GID.
            machine.succeed(f"groupdel --force {group}")

    exercise_order(controlFirst, "control")
    exercise_order(workerFirst, "worker")

    # Model an older shared-identity installation, then the explicit stopped
    # operator migration in deployment.md. No implicit unit or state rewrite.
    controlFirst.succeed(
        f"systemctl stop astrumweaver-control.service astrumweaver-worker.service; "
        f"cp {control_unit} /tmp/reviewed-control.unit; cp {worker_unit} /tmp/reviewed-worker.unit; "
        f"sed -i -e '/^SupplementaryGroups=/d' -e 's/^User=.*/User=astrumweaver/' "
        f"-e 's/^Group=.*/Group=astrumweaver/' -e 's/^StateDirectory=.*/StateDirectory=astrumweaver/' {control_unit}; "
        f"sed -i '/^SupplementaryGroups=/d' {worker_unit}; "
        f"{chown} root:astrumweaver {' '.join(control_files)}"
    )
    reject(controlFirst, "legacy-control-retry", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control", "astrumweaver"], needle="legacy/shared")
    reject(controlFirst, "legacy-worker-retry", worker_cmd("astrumweaver"),
           users=["astrumweaver-control", "astrumweaver"], needle="legacy/shared")
    controlFirst.succeed(
        f"{chown} root:root {' '.join(control_files)}; {chmod} 0600 {' '.join(control_files)}; "
        f"cp /tmp/reviewed-control.unit {control_unit}; cp /tmp/reviewed-worker.unit {worker_unit}"
    )
    snapshot(controlFirst, "worker-state-preserved", paths=[worker_state, worker_state + "/permission-write-check"])
    install(controlFirst, "control", "astrumweaver-control")
    check_snapshot(controlFirst, "worker-state-preserved", paths=[worker_state, worker_state + "/permission-write-check"])
    install(controlFirst, "worker", "astrumweaver")
    assert_all(controlFirst, "astrumweaver-control", "astrumweaver")
    assert_access(controlFirst, "astrumweaver-control", "astrumweaver")
    assert_systemd_access(controlFirst, "astrumweaver-control", "astrumweaver")

    prepare(customIdentity)
    customIdentity.succeed(
        "getent group custom-control && getent group custom-worker && "
        "! getent passwd custom-control && ! getent passwd custom-worker"
    )
    customIdentity.succeed("getent group custom-control > /tmp/pre-control-group; getent group custom-worker > /tmp/pre-worker-group")
    install(customIdentity, "control", "custom-control")
    install(customIdentity, "worker", "custom-worker")
    customIdentity.succeed("getent group custom-control > /tmp/post-control-group; getent group custom-worker > /tmp/post-worker-group; "
                           f"{cmp} /tmp/pre-control-group /tmp/post-control-group; {cmp} /tmp/pre-worker-group /tmp/post-worker-group")
    assert_all(customIdentity, "custom-control", "custom-worker")
    install(customIdentity, "control", "custom-control")
    install(customIdentity, "worker", "custom-worker")
    assert_all(customIdentity, "custom-control", "custom-worker")
    assert_access(customIdentity, "custom-control", "custom-worker")
    assert_systemd_access(customIdentity, "custom-control", "custom-worker")

    clear(customIdentity)
    install(customIdentity, "control", "same-role")
    reject(customIdentity, "same-account", worker_cmd("same-role"), users=["same-role"], needle="legacy/shared")

    prepare(negativeFixtures)
    clear(negativeFixtures)
    reject(negativeFixtures, "wrong-primary", control_cmd("bad-primary"),
           users=["bad-primary", "astrumweaver-control"])

    clear(negativeFixtures)
    install(negativeFixtures, "worker", "cross-worker")
    negativeFixtures.succeed("gpasswd --add cross-control cross-worker")
    reject(negativeFixtures, "supplementary-control", control_cmd("cross-control"),
           users=["cross-control", "cross-worker"])
    negativeFixtures.succeed("gpasswd --delete cross-control cross-worker")

    # Reverse direction: Worker is a member of the Control private group.
    clear(negativeFixtures)
    install(negativeFixtures, "control", "cross-control")
    negativeFixtures.succeed("gpasswd --add cross-worker cross-control")
    reject(negativeFixtures, "supplementary-worker", worker_cmd("cross-worker"),
           users=["cross-control", "cross-worker"])
    negativeFixtures.succeed("gpasswd --delete cross-worker cross-control")

    clear(negativeFixtures)
    negativeFixtures.succeed(
        f"{mkdir} -p /etc/astrumweaver {worker_state}; printf sentinel > /etc/astrumweaver/unchanged; "
        f"{chmod} 0700 /etc/astrumweaver; {chmod} 0701 {worker_state}"
    )
    reject(negativeFixtures, "legacy-state", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control"], needle="legacy/shared", paths=sentinel_paths)

    clear(negativeFixtures)
    negativeFixtures.succeed(
        f"{mkdir} -p /etc/astrumweaver /etc/systemd/system {control_state}; "
        f"printf sentinel > /etc/astrumweaver/unchanged; "
        f"printf '[Service]\\nUser=astrumweaver\\nGroup=astrumweaver\\nStateDirectory=astrumweaver\\n' > {control_unit}; "
        f"{chmod} 0700 /etc/astrumweaver {control_state}; {chmod} 0600 {control_unit}"
    )
    reject(negativeFixtures, "legacy-unit", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control"], needle="legacy/shared", paths=sentinel_paths)

    clear(negativeFixtures)
    negativeFixtures.succeed(
        f"{mkdir} -p /etc/astrumweaver /etc/systemd/system; "
        f"printf '[Service]\\nUser=astrumweaver-control\\nGroup=astrumweaver-control\\n' > {control_unit}"
    )
    reject(negativeFixtures, "malformed-unit", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control"])

    clear(negativeFixtures)
    negativeFixtures.succeed(
        f"{mkdir} -p /etc/astrumweaver /etc/systemd/system; "
        f"printf '[Service]\\nUser=astrumweaver-control\\nGroup=other-service\\nStateDirectory=astrumweaver-control\\n' > {control_unit}"
    )
    reject(negativeFixtures, "group-mismatch", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control"])

    clear(negativeFixtures)
    negativeFixtures.succeed(
        f"{mkdir} -p /etc/astrumweaver /etc/systemd/system/astrumweaver-worker.service.d; "
        f"printf '[Service]\\nSupplementaryGroups=other-service\\n' > {dropin_override}"
    )
    reject(negativeFixtures, "dropin-override", control_cmd("astrumweaver-control"),
           users=["astrumweaver-control"])

    for index, role in enumerate(["control", "worker"], 1):
        clear(negativeFixtures)
        user = f"numeric-{role}-uid"
        alias = f"{user}-alias"
        negativeFixtures.succeed(f"groupadd --gid {41000 + index} {user}; "
                                 f"useradd --uid {41100 + index} --gid {user} --no-create-home --home-dir /var/empty {user}; "
                                 f"useradd --non-unique --uid {41100 + index} --gid nogroup --no-create-home --home-dir /var/empty {alias}")
        reject_numeric(negativeFixtures, f"own-{role}-uid-alias", role, user, "UID",
                       users=[user, alias], groups=[user, "astrumweaver-config"])
        remove_fixture_alias(negativeFixtures, user=alias)

    for index, role in enumerate(["control", "worker"], 1):
        clear(negativeFixtures)
        user = f"numeric-{role}-gid"
        alias = f"{user}-alias"
        negativeFixtures.succeed(f"groupadd --gid {42000 + index} {user}; "
                                 f"groupadd --non-unique --gid {42000 + index} {alias}; "
                                 f"useradd --uid {42100 + index} --gid {user} --no-create-home --home-dir /var/empty {user}; "
                                 f"gpasswd --add nobody {alias}; test -z \"$(getent group {user} | {cut} -d: -f4)\"")
        reject_numeric(negativeFixtures, f"own-{role}-gid-alias", role, user, "GID",
                       users=[user], groups=[user, alias, "astrumweaver-config"])
        remove_fixture_alias(negativeFixtures, group=alias)

    for index, role in enumerate(["control", "worker"], 1):
        clear(negativeFixtures)
        user = f"numeric-{role}-absent"
        alias = f"{user}-alias"
        negativeFixtures.succeed(f"groupadd --gid {43000 + index} {user}; "
                                 f"groupadd --non-unique --gid {43000 + index} {alias}; "
                                 f"gpasswd --add nobody {alias}; test -z \"$(getent group {user} | {cut} -d: -f4)\"; "
                                 f"! getent passwd {user}")
        reject_numeric(negativeFixtures, f"absent-{role}-gid-alias", role, user, "GID",
                       users=[user], groups=[user, alias, "astrumweaver-config"])
        remove_fixture_alias(negativeFixtures, group=alias)

    clear(negativeFixtures)
    config_gid = negativeFixtures.succeed(
        f"getent group astrumweaver-config | {cut} -d: -f3"
    ).strip()
    config_alias = "numeric-config-gid-alias"
    negativeFixtures.succeed(f"groupadd --non-unique --gid {config_gid} {config_alias}; "
                             f"gpasswd --add nobody {config_alias}")
    for role in ["control", "worker"]:
        clear(negativeFixtures)
        user = f"numeric-config-{role}"
        reject_numeric(negativeFixtures, f"shared-config-{role}-gid-alias", role, user, "GID",
                       users=[user], groups=["astrumweaver-config", config_alias])
    remove_fixture_alias(negativeFixtures, group=config_alias)

    # Exercise the installed peer path as well as the requested role path.
    for index, kind in enumerate(["UID", "GID"], 1):
        clear(negativeFixtures)
        peer = f"numeric-peer-{kind.lower()}-worker"
        user = f"numeric-peer-{kind.lower()}-control"
        alias_group = f"numeric-peer-{kind.lower()}-alias"
        install(negativeFixtures, "worker", peer)
        if kind == "UID":
            alias = f"{peer}-alias"
            peer_uid = negativeFixtures.succeed(f"getent passwd {peer} | {cut} -d: -f3").strip()
            negativeFixtures.succeed(f"groupadd --gid {44000 + index} {alias_group}; "
                                     f"useradd --non-unique --uid {peer_uid} --gid {alias_group} --no-create-home --home-dir /var/empty {alias}")
            alias_users = [peer, user, alias]
        else:
            peer_gid = negativeFixtures.succeed(f"getent group {peer} | {cut} -d: -f3").strip()
            negativeFixtures.succeed(f"groupadd --non-unique --gid {peer_gid} {alias_group}; "
                                     f"gpasswd --add nobody {alias_group}")
            alias_users = [peer, user]
        reject_numeric(negativeFixtures, f"peer-{kind.lower()}-alias", "control", user, kind,
                       users=alias_users,
                       groups=["astrumweaver-config", peer, alias_group])
        if kind == "UID":
            remove_fixture_alias(negativeFixtures, user=alias, group=alias_group)
        else:
            remove_fixture_alias(negativeFixtures, group=alias_group)
  '';
}
