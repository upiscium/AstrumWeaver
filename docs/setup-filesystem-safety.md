# Privileged setup filesystem boundary

Generic-systemd `SystemdSetupDriver` keeps installer authority separate from
Worker data. This is the F4 fix tracked by #81 / release review #79.

## Locations and authority

- `/var/lib/astrumweaver-setup` is root-owned, mode `0700`. Per-action receipts
  are root-owned regular files, mode `0600`, below this directory.
- `/var/lib/astrumweaver` remains the Worker's writable home/state. Runtime
  state/cache directories remain service-owned with mode `0750`.
- `/etc/astrumweaver/runtime` contains root-managed runtime configuration and
  the NVIDIA driver bridge. Configuration files retain Worker-group read access.
- The default TabbyAPI configuration is now
  `/etc/astrumweaver/runtime/exllamav3/config.yml`, not Worker-writable state.
  An explicitly configured alternative must also have safe root-owned ancestry.

Legacy `packages/*.installed` and `model-preparation/*` markers below Worker
state are neither trusted nor migrated. They can remain in place without being
read or written by the new bookkeeping path. A receipt is only a record of a
completed action; its existence never proves that a prerequisite still exists.

## Prerequisite checks

For known single-command providers, package inspection checks the current
runtime command. Package installation must make that check pass before a
completion receipt is written. This is availability checking, not a claim that
CUDA, the model, or runtime health has already passed the later startup gate.

A custom/package-group installation (including ExLlamaV3) and native remote
model preparation need an explicit read-only verifier when no built-in check
can establish the prerequisite. Configure a provider-keyed argv map alongside
the existing installer/downloader/converter maps:

```sh
export ASTRUMWEAVER_RUNTIME_VERIFIERS_JSON='{"exllamav3":["/usr/local/sbin/verify-reviewed-exllamav3"]}'
```

The driver appends `package <package_reference>` or `model <model_ref>` to this
prefix, without a shell. The verifier must check the selected installation or
prepared model using deployment-owned configuration and return:

- `0`: prerequisite currently exists and is usable according to the check;
- `1`: prerequisite is absent, so its reviewed preparation is needed;
- any other exit status, an execution error or a 10-second timeout: blocked.

Verifiers are operator-reviewed, read-only commands. They may run during
planning/dry-run and must never install, download, convert or repair anything.
They must not merely check a completion marker. Their output is discarded and
is not copied to public evidence. Existing package/model commands still run
only during approved apply. Preparation is checked again after those commands
return; exit code zero from an installer alone cannot create a success receipt.

An absolute local download destination can be checked for existence. Conversion
always needs a verifier for the converted output: existence of the source
model is not evidence that conversion completed. Ordinary local model
`reference_only` selection does not need an additional verifier.

## Unsafe files and recovery

Authoritative files and their parent directories are opened without following
symlinks, using directory descriptors. Regular/dangling links, multiply-linked
files, non-regular files, untrusted owners and group/world-writable ancestors
are refused. A root-owned sticky system temporary ancestor may be traversed,
but every descendant is still checked. Staged roots are owned by the invoking
user and their ancestry is checked too; staging never bypasses link checks.

New files use exclusive random temporary names and descriptor-relative atomic
replacement. Configuration rollback uses the same checks and refuses to
replace an object whose contents changed after apply. A failure does not
silently adopt an existing object by changing its owner or permissions.

If setup is blocked by old or operator-managed state, stop at the reported
boundary. Inspect the affected ownership/path locally and make an explicit
recovery decision; do not recursively chmod/chown a Worker-writable tree to
turn it into trusted installer state, and do not copy legacy markers into the
new directory. Preserve unrelated files and the running deployment. An existing
TabbyAPI deployment using the former writable config path needs a separately
reviewed migration; this change does not rewrite it behind the operator's back.

## Validation and release boundary

Regression tests cover linked targets/ancestors, wrong ownership/permissions,
forged/stale markers, temporary-name collisions and replacement races, partial
render failure, safe rollback and ordinary idempotent setup. These regressions
are not real-host release acceptance. #79 stays HOLD until its other blockers
and candidate checks pass; #80 remains the separate documentation-only Real
Smoke requiring upiscium's explicit acceptance.
