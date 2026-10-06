import pytest

from orbit.config import parse_limit


def test_parse_limit_positive_integer():
    assert parse_limit("12") == 12


def test_parse_limit_rejects_negative():
    with pytest.raises(ValueError):
        parse_limit("-1")