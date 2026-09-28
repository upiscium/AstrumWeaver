"""Exercise the real identity validator with controlled NSS enumeration."""

import os
from pathlib import Path
import subprocess

import pytest


COMMON = Path(__file__).resolve().parents[1] / "setup/lib/common.sh"


def validate(database: str, records: str, *, status: int = 0):
    return subprocess.run(
        [
            "bash", "-c", r'''
source "$1"
getent() {
  [[ "$#" == 1 && "$1" == "$IDENTITY_DATABASE" ]] || return 99
  printf '%s\n' "$IDENTITY_RECORDS"
  return "$ENUMERATION_STATUS"
}
validate_unique_numeric_id "$2" role-service 1200
''',
            "identity-test", str(COMMON), database,
        ],
        env={
            **os.environ,
            "IDENTITY_DATABASE": database,
            "IDENTITY_RECORDS": records,
            "ENUMERATION_STATUS": str(status),
        },
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("database,kind", [("passwd", "UID"), ("group", "GID")])
@pytest.mark.parametrize("alias_first", [False, True])
@pytest.mark.parametrize("alias_number", ["1200", "01200"])
def test_aliases_are_rejected_regardless_of_name_order_or_spelling(
    database, kind, alias_first, alias_number,
):
    own = "role-service:x:1200:"
    alias = f"unrelated-alias:private-password-field:{alias_number}:alice"
    records = [alias, own] if alias_first else [own, alias]
    result = validate(database, "\n".join(records))
    assert result.returncode != 0
    assert f"numeric {kind} alias" in result.stderr
    assert "private-password-field" not in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("database", ["passwd", "group"])
def test_unique_id_allows_other_ids(database):
    result = validate(database, "root:x:0:\nrole-service:x:1200:\nalice:x:1201:")
    assert result.returncode == 0, result.stderr


def test_uid_is_not_compared_to_primary_gid_namespace():
    result = validate(
        "passwd",
        "role-service:x:1200:1300:Role:/var/empty:/bin/false\n"
        "alice:x:1300:1200:Other:/var/empty:/bin/false",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("database", ["passwd", "group"])
@pytest.mark.parametrize("status,records", [
    (2, "role-service:x:1200:"),  # Even partial successful-looking output fails.
    (3, ""),
    (0, ""),
    (0, "alice:x:1201:"),  # Enumeration must confirm the keyed lookup.
    (0, "role-service:x:1200:\nmalformed:x:not-an-id:"),
])
def test_failed_or_inconsistent_enumeration_fails_closed(database, status, records):
    result = validate(database, records, status=status)
    assert result.returncode != 0
    assert "refusing setup" in result.stderr
