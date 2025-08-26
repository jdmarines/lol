"""Coloca aquí tu lógica real para descargar campeones de DataDragon y guardar name2id."""
import json, os

def save_name2id(output_path: str, mapping: dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    dummy = {
        "Aatrox": 266, "Ahri": 103, "Akali": 84, "Alistar": 12, "Amumu": 32,
        "Wukong": 62, "Gnar": 150, "Jinx": 222, "Lulu": 117, "Vi": 254, "Yone": 777,
        "Rumble": 68, "Azir": 268, "Braum": 201, "Aurora": 893
    }
    save_name2id("data/artifacts/name2id.json", dummy)
    print("Escribí data/artifacts/name2id.json (dummy). Reemplázame con la descarga real.")
