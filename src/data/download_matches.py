"""Coloca aquí tu lógica real para descargar partidas y construir id2idx."""
import json, os

def save_id2idx(output_path: str, mapping: dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    dummy = {
        "266": 0, "103": 1, "84": 2, "12": 3, "32": 4, "62": 5, "150": 6,
        "222": 7, "117": 8, "254": 9, "777": 10, "68": 11, "268": 12, "201": 13, "893": 14
    }
    save_id2idx("data/artifacts/id2idx.json", dummy)
    print("Escribí data/artifacts/id2idx.json (dummy). Reemplázame con tu índice real.")
