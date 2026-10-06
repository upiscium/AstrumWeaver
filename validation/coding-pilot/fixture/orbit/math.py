def normalize_score(value: float) -> float:
    return round(value / 100.0, 4)


def clamp_score(value: int) -> int:
    return max(0, min(100, value))