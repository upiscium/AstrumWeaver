from .math import normalize_score


def normalized_payload(raw_score: float) -> dict[str, float]:
    return {"score": normalize_score(raw_score)}