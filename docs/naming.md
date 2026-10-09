# Naming and compatibility: TSUMGI

**TSUMGI（紡ぎ）** is the public project and GitHub repository name, replacing **AstrumWeaver** as of the October 2026 naming decision ([issue #120](https://github.com/upiscium/TSUMGI/issues/120)).

TSUMGI is a deployment-agnostic compute fabric. The project rename does **not** constitute an upgrade or migration of any running Control Plane or Worker. In particular, the following lower-case names and uppercase configuration prefixes are **still required by the current v0.1 software**.

| Surface | Current identity / interface | Phase A policy |
| --- | --- | --- |
| GitHub project and docs | `upiscium/TSUMGI` | New canonical name |
| Python distribution | `astrumweaver` | Keep unchanged |
| Python import namespace | `astrumweaver` | Keep unchanged |
| Installed command prefix | `astrumweaver-*` | Keep unchanged |
| Nix flake package output | `.#astrumweaver` | Keep unchanged |
| NixOS options | `services.astrumweaver.control`, `services.astrumweaver.worker` | Keep unchanged |
| systemd unit names | `astrumweaver-control.service`, `astrumweaver-worker.service` | Keep unchanged |
| Service users and groups | `astrumweaver`, `astrumweaver-control`, `astrumweaver-config` | Keep unchanged |
| Environment variables | `ASTRUMWEAVER_*` | Keep unchanged |
| Configuration and state paths | `/etc/astrumweaver`, `/var/lib/astrumweaver`, `/var/lib/astrumweaver-control` | Keep unchanged |
| Managed runtime Nix profiles | `/nix/var/nix/profiles/astrumweaver-*` | Keep unchanged |
| Existing database names and credentials | Site/operator-configured | Do not change |
| API and protocol compatibility | Existing endpoints, schemas and client expectations | Keep unchanged |

## Source checkouts and references

The canonical GitHub remote after the repository rename is:

```sh
git remote set-url origin https://github.com/upiscium/TSUMGI.git
git remote -v
```

Run this **in each chosen local checkout**, after verifying which checkout is being updated. GitHub generally redirects old repository URLs, but redirects are not a substitute for updating hard-coded CI integrations, submodules, deployment automation, webhooks, references, or flake inputs.

For new Nix flake installation commands, use `github:upiscium/TSUMGI`. Existing pinned `github:upiscium/AstrumWeaver` references must be inventoried and migrated at their owners' discretion; do not rewrite locks or trigger fleet-wide rebuilds as part of rebranding.

## Deliberate compatibility boundary

An existing installation must continue using its current Python imports, CLI executables, NixOS option keys, service names, users, environment variables, and on-disk layout. Do **not** manually rename a systemd unit, move `/etc` or `/var/lib` directories, change PostgreSQL names, or change ownership simply to match the new project name.

Changes to user-visible project descriptions are safe to apply independently. Renaming operational identifiers must be designed, tested and accepted separately; such a change requires explicit aliases, fresh-install and upgrade coverage, mixed-version Control/Worker validation, a rollback plan, and staged GPU-worker acceptance.

The staged technical migration and outstanding checks are tracked in [#120](https://github.com/upiscium/TSUMGI/issues/120).
