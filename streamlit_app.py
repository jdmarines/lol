
import os
import streamlit as st
from typing import List

from src.models.torch_model import load_model, load_name2id, predict_proba_names, normalize_team

st.set_page_config(page_title="LoL Win Probability (PyTorch)", page_icon="🧠", layout="centered")
st.title("🧠 LoL — Probabilidad de Victoria (modelo PyTorch)")
st.caption("Ingresa 5 campeones por equipo. El modelo Bradley–Terry calcula P(Blue) y P(Red).")

with st.expander("⚙️ Configuración de rutas"):
    default_model = os.environ.get("MODEL_PATH", "src/models/checkpoints/bt_transformer_model.pt")
    default_name2id = os.environ.get("NAME2ID_PATH", "data/artifacts/name2id.json")
    default_champs = os.environ.get("CHAMPS_CSV", "data/download_champions.csv")
    model_path = st.text_input("MODEL_PATH", default_model)
    name2id_path = st.text_input("NAME2ID_PATH (opcional)", default_name2id)
    champs_csv = st.text_input("CHAMPIONS CSV (fallback si no hay NAME2ID)", default_champs)

    if st.button("Cargar modelo y mappings"):
        try:
            st.session_state.model, st.session_state.id2idx = load_model(model_path, map_location="cpu")
            st.session_state.name2id = load_name2id(name2id_path if os.path.exists(name2id_path) else None,
                                                    champs_csv if os.path.exists(champs_csv) else None)
            st.success("Modelo y mappings cargados.")
        except Exception as e:
            st.exception(e)

def team_inputs(prefix: str, key: str) -> List[str]:
    cols = st.columns(5)
    champs = []
    for i, c in enumerate(cols):
        champs.append(c.text_input(f"{prefix} {i+1}", key=f"{key}_{i}"))
    champs = [x.strip() for x in champs if x.strip()]
    return champs

st.subheader("Equipo Azul")
blue_team = team_inputs("Campeón", "blue")
st.subheader("Equipo Rojo")
red_team = team_inputs("Campeón", "red")

st.divider()

if st.button("Calcular probabilidades"):
    if "model" not in st.session_state:
        st.warning("Primero carga el modelo en la sección de configuración.")
        st.stop()
    # Normalización visible
    n_blue = normalize_team(blue_team)
    n_red  = normalize_team(red_team)
    if n_blue != blue_team or n_red != red_team:
        st.info(f"Nombres normalizados:\n- Azul: {n_blue}\n- Rojo: {n_red}")

    try:
        p_blue, p_red = predict_proba_names(st.session_state.model, st.session_state.id2idx, st.session_state.name2id, n_blue, n_red)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success("¡Cálculo completado!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("P(Blue gana)", f"{p_blue:.3f}")
    with col2:
        st.metric("P(Red gana)", f"{p_red:.3f}")
