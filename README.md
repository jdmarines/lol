# LoL Team Win Probability — Streamlit App

Pequeña app de Streamlit para **ingresar 5 campeones por equipo** y calcular las **probabilidades de victoria** usando tu embedding/modelo existente.

## Estructura

```
.
├─ streamlit_app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
├─ data/
│  └─ artifacts/
│     ├─ name2id.json          # requerido
│     ├─ id2idx.json           # requerido
│     └─ README.md
├─ models/
│  └─ checkpoints/
│     └─ model.pkl             # requerido (tu modelo entrenado)
└─ src/
   ├─ data/
   │  ├─ download_champions.py
   │  └─ download_matches.py
   ├─ models/
   │  ├─ embedding.py
   │  └─ predict.py
   └─ utils/
      └─ names.py
```

> ⚠️ **Coloca tus artefactos reales** en:
> - `data/artifacts/name2id.json`
> - `data/artifacts/id2idx.json`
> - `models/checkpoints/model.pkl` (puede ser un `joblib`/`pickle` de sklearn/torch wrapper).

## Instalación local

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Despliegue rápido en Streamlit Community Cloud

1. Sube este repo a GitHub.
2. En streamlit.io, crea una nueva app apuntando al repo y `streamlit_app.py`.
3. En *Secrets* (⚙️), agrega rutas o credenciales si las necesitas. Si tus artefactos son livianos (<100 MB), pueden ir versionados en el repo. Si pesados, súbelos a almacenamiento (S3/GDrive) y modifica `load_artifacts()` para descargarlos al inicio.

## Endpoints/Funciones clave

- `src/utils/names.py`: normalización de nombres (e.g., `MonkeyKing` → `Wukong`, remover `BigGnar`). 
- `src/models/predict.py`: `predict_proba_names(...)` que carga artefactos y arma el vector 10×ids → embedding → proba.
- `streamlit_app.py`: UI, validación y visualización.

## Notas

- La app espera exactamente **5 campeones por lado** y todos distintos dentro del mismo equipo.
- Si falta algún artefacto, la app mostrará una advertencia clara.


## Uso con tu modelo PyTorch existente

1. Coloca tu archivo **`bt_transformer_model.pt`** en `models/checkpoints/`.
2. Provee un mapping `data/artifacts/name2id.json` **o** un `data/champions.csv` para derivarlo.
3. Ejecuta:
   ```bash
   streamlit run streamlit_app.py
   ```
4. En la expander de configuración, presiona **Cargar modelo y mappings**.
