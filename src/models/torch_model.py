
import os, json
from typing import List, Tuple, Dict, Optional

import torch
import pandas as pd

# --- Normalización de nombres ---
NORMALIZATION_MAP = {
    "MonkeyKing": "Wukong",
    "BigGnar": "Gnar",
    "biggnar": "Gnar",
    "big-gnar": "Gnar",
}

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    return NORMALIZATION_MAP.get(name.strip(), name.strip())

def normalize_team(team: List[str]) -> List[str]:
    return [normalize_name(x) for x in team]

# --- Carga de name2id ---
def load_name2id(name2id_path: Optional[str], champions_csv_path: Optional[str]) -> Dict[str, int]:
    # 1) JSON explícito (si existe)
    if name2id_path and os.path.exists(name2id_path):
        with open(name2id_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # asegurar ints
        return {k: int(v) for k, v in d.items()}
    # 2) champions.csv (apiid, apiname, name)
    if champions_csv_path and os.path.exists(champions_csv_path):
        df = pd.read_csv(champions_csv_path)
        # Columnas posibles
        id_col = "apiid" if "apiid" in df.columns else ("id" if "id" in df.columns else None)
        name_col = "apiname" if "apiname" in df.columns else ("name" if "name" in df.columns else None)
        if id_col and name_col:
            m = dict(zip(df[name_col].astype(str), df[id_col].astype(int)))
            return m
    raise FileNotFoundError("No se pudo cargar name2id: provee data/artifacts/name2id.json o data/champions.csv")

# --- Carga de checkpoint ---
def load_model(model_path: str, map_location: str = "cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en {model_path}")
    ckpt = torch.load(model_path, map_location=map_location)
    # El checkpoint debe contener 'model_state' y metadatos para reconstruir arquitectura mínima
    # Para inferencia, solo necesitamos parámetros para la parte forward sA-sB (Bradley–Terry).
    # Reconstruimos una clase minimal para cargar pesos exactamente.
    config = ckpt.get("config", {})
    emb_dim = ckpt.get("emb_dim", 32)
    id2idx = ckpt.get("id2idx", None)
    if id2idx is None:
        raise ValueError("El checkpoint no tiene 'id2idx'. Reentrena guardando ese campo.")

    # Modelo mínimo equivalente
    import torch.nn as nn

    class TeamEncoder(nn.Module):
        def __init__(self, d_model=emb_dim, nhead=config.get("nhead", 4), num_layers=config.get("num_layers", 2)):
            super().__init__()
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.slot_emb = nn.Embedding(5, d_model)

        def forward(self, tokens):
            B, S, D = tokens.shape
            device = tokens.device
            slots = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
            x = tokens + self.slot_emb(slots)
            h = self.encoder(x)
            z = h.mean(dim=1)
            return z

    class BTModel(nn.Module):
        def __init__(self, num_champs, d_model=emb_dim, nhead=config.get("nhead", 4), num_layers=config.get("num_layers", 2)):
            super().__init__()
            self.embed = nn.Embedding(num_champs, d_model)
            self.team = TeamEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model//2), nn.ReLU(),
                nn.Linear(d_model//2, 1)
            )

        def team_score(self, idxs):
            tok = self.embed(idxs)
            z = self.team(tok)
            s = self.head(z).squeeze(-1)
            return s

        def forward(self, b_idx, r_idx):
            sA = self.team_score(b_idx)
            sB = self.team_score(r_idx)
            logits = sA - sB
            return logits

    # Reconstruimos tamaño de vocab a partir de id2idx
    num_champs = max(int(i) for i in id2idx.values()) + 1
    model = BTModel(num_champs=num_champs, d_model=emb_dim)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, id2idx

# --- Predicción con nombres ---
@torch.no_grad()
def predict_proba_names(
    model, id2idx: Dict[int,int], name2id: Dict[str,int],
    blue_team: List[str], red_team: List[str]
) -> Tuple[float, float]:
    import torch
    # normalizar y mapear
    blue_team = normalize_team(blue_team)
    red_team  = normalize_team(red_team)

    if len(blue_team) != 5 or len(red_team) != 5:
        raise ValueError("Se requieren exactamente 5 campeones por equipo.")
    if len(set(blue_team)) != 5 or len(set(red_team)) != 5:
        raise ValueError("Campeones repetidos dentro de un mismo equipo.")

    def to_idx(names):
        ids = [int(name2id[n]) for n in names]
        idxs = [int(id2idx[i]) for i in ids]
        return torch.tensor([idxs], dtype=torch.long)

    b_idx = to_idx(blue_team)
    r_idx = to_idx(red_team)
    logits = model(b_idx, r_idx)
    p_blue = torch.sigmoid(logits).item()
    return float(p_blue), float(1.0 - p_blue)
