from typing import List, Dict
import numpy as np

def compose_match_vector(blue_ids: List[int], red_ids: List[int], id2idx: Dict[int, int]) -> np.ndarray:
    """Crea el vector de entrada a partir de 5 ids azules + 5 ids rojos.
    Esquema por defecto: bolsa de campeones con signo (+1 azul, -1 rojo).
    Ajusta si tu embedding real usa otra representación.
    """
    dim = max(id2idx.values()) + 1 if id2idx else 0
    x = np.zeros(dim, dtype=np.float32)
    for cid in blue_ids:
        if cid in id2idx:
            x[id2idx[cid]] += 1.0
    for cid in red_ids:
        if cid in id2idx:
            x[id2idx[cid]] -= 1.0
    return x.reshape(1, -1)
