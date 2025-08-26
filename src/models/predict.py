import json
import os
from typing import List, Dict, Tuple
import numpy as np
from joblib import load

from src.utils.names import normalize_team
from src.models.embedding import compose_match_vector

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "data/artifacts")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/checkpoints/model.pkl")

def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_artifacts() -> Tuple[Dict[str, int], Dict[int, int], object]:
    name2id_path = os.path.join(ARTIFACTS_DIR, "name2id.json")
    id2idx_path = os.path.join(ARTIFACTS_DIR, "id2idx.json")

    if not os.path.exists(name2id_path):
        raise FileNotFoundError(f"Falta {name2id_path}.")
    if not os.path.exists(id2idx_path):
        raise FileNotFoundError(f"Falta {id2idx_path}.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Falta el modelo en {MODEL_PATH}.")

    name2id = _load_json(name2id_path)
    _id2idx = _load_json(id2idx_path)
    id2idx = {int(k): int(v) for k, v in _id2idx.items()}

    model = load(MODEL_PATH)
    return name2id, id2idx, model

def names_to_ids(names: List[str], name2id: Dict[str, int]) -> List[int]:
    return [name2id[n] for n in names]

def predict_proba_names(blue_team: List[str], red_team: List[str]):
    blue_team = normalize_team(blue_team)
    red_team = normalize_team(red_team)

    name2id, id2idx, model = load_artifacts()
    try:
        blue_ids = names_to_ids(blue_team, name2id)
        red_ids = names_to_ids(red_team, name2id)
    except KeyError as e:
        raise KeyError(f"Campeón desconocido: {e.args[0]}. Verifica nombre y capitalización.")

    X = compose_match_vector(blue_ids, red_ids, id2idx)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # Asumimos proba para clase 'blue gana' en índice 1 si binaria
        if hasattr(proba, "__len__") and len(proba) == 2:
            p_blue = float(proba[1])
        else:
            p_blue = float(proba[0])
        p_red = 1.0 - p_blue
        return p_blue, p_red

    if hasattr(model, "predict"):
        y = model.predict(X)
        import numpy as np
        if isinstance(y, (list, tuple, np.ndarray)):
            y = float(np.ravel(y)[0])
        y = float(max(0.0, min(1.0, y)))
        return y, 1.0 - y

    raise ValueError("El modelo no tiene 'predict_proba' ni 'predict'. Ajusta predict_proba_names.")
