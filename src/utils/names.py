from typing import List

NORMALIZATION_MAP = {
    "MonkeyKing": "Wukong",
    "BigGnar": "Gnar",
    "biggnar": "Gnar",
    "big-gnar": "Gnar",
}

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    n = name.strip()
    return NORMALIZATION_MAP.get(n, n)

def normalize_team(team: List[str]) -> List[str]:
    return [normalize_name(x) for x in team]
