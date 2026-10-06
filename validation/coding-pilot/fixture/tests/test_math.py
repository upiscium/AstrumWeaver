from orbit.math import clamp_score, normalize_score


def test_normalize_score():
    assert normalize_score(50) == 0.5


def test_clamp_score_inside_range():
    assert clamp_score(42) == 42