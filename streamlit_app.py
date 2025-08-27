
import os
import streamlit as st
from typing import List

from src.models.torch_model import load_model, load_name2id, predict_proba_names, normalize_team

st.set_page_config(page_title="LoL Win Probability (PyTorch)", page_icon="🧠", layout="centered")

# --- Styles ---
st.markdown("""
<style>
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:13px;
}
.badge-blue { background: rgba(33,150,243,.15); color:#90caf9; border:1px solid rgba(33,150,243,.35); }
.badge-red  { background: rgba(244,67,54,.15); color:#ef9a9a; border:1px solid rgba(244,67,54,.35); }
.panel {
  padding: 14px 14px 6px; border-radius:14px; border:1px solid var(--border);
  margin-bottom: 10px;
}
.panel-blue { border-color: rgba(33,150,243,.35); background: rgba(33,150,243,.07); }
.panel-red  { border-color: rgba(244,67,54,.35); background: rgba(244,67,54,.07); }
hr.soft { border:0; height:1px; background: rgba(255,255,255,.08); margin: 12px 0; }
.small { font-size: 12px; opacity:.75; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 LoL — Probabilidad de Victoria (modelo PyTorch)")
st.caption("Selecciona 5 campeones por equipo en sus roles. El modelo Bradley–Terry calcula P(Blue) y P(Red).")

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

ROLE_LABELS = ["Top Laner", "Jungla", "Mid Laner", "ADC", "Soporte"]

def team_selects(side_label: str, side_key: str, panel_class: str, badge_class: str) -> List[str]:
    st.markdown(f'<div class="panel {panel_class}">'
                f'<span class="badge {badge_class}">{side_label}</span>'
                f'<hr class="soft"/></div>', unsafe_allow_html=True)
    container = st.container()
    options = sorted(st.session_state.get("name2id", {}).keys())
    selections = []
    used = set()
    cols = container.columns(5)
    for i, c in enumerate(cols):
        label = ROLE_LABELS[i]
        if options:
            opts = [""] + [o for o in options if o not in used]
            sel = c.selectbox(label, options=opts, index=0, key=f"{side_key}_sel_{i}", placeholder=f"Selecciona {label.lower()}")
        else:
            sel = c.text_input(label, key=f"{side_key}_txt_{i}")
        sel = sel.strip() if isinstance(sel, str) else sel
        if sel:
            selections.append(sel); used.add(sel)
        else:
            selections.append("")
    # inline validation hints
    filled = [s for s in selections if s]
    if len(filled) < 5:
        container.markdown('<span class="small">Selecciona los 5 campeones para este equipo.</span>', unsafe_allow_html=True)
    if len(filled) != len(set(filled)):
        container.markdown('<span class="small" style="color:#ef9a9a;">No se pueden repetir campeones dentro del mismo equipo.</span>', unsafe_allow_html=True)
    return selections

blue_team = team_selects("🔵 Equipo Azul", "blue", "panel-blue", "badge-blue")
red_team  = team_selects("🔴 Equipo Rojo", "red",  "panel-red",  "badge-red")

st.divider()

def validate_teams(blue: List[str], red: List[str]) -> bool:
    if any(x == "" for x in blue) or any(x == "" for x in red):
        st.warning("Debes seleccionar **5 campeones por equipo**.")
        return False
    if len(set(blue)) != 5:
        st.error("Hay campeones repetidos en el **equipo azul**.")
        return False
    if len(set(red)) != 5:
        st.error("Hay campeones repetidos en el **equipo rojo**.")
        return False
    overlap = set(blue).intersection(set(red))
    if overlap:
        st.error(f"No se puede seleccionar el mismo campeón en ambos equipos: {sorted(overlap)}")
        return False
    return True

if st.button("Calcular probabilidades"):
    if "model" not in st.session_state:
        st.warning("Primero carga el modelo en la sección de configuración.")
        st.stop()
    if "name2id" not in st.session_state:
        st.warning("No se han cargado los mappings de campeones (name2id).")
        st.stop()

    if not validate_teams(blue_team, red_team):
        st.stop()

    # Normalización visible
    n_blue = normalize_team(blue_team)
    n_red  = normalize_team(red_team)
    if n_blue != blue_team or n_red != red_team:
        st.info(f"Nombres normalizados:\n- Azul: {n_blue}\n- Rojo: {n_red}")

    try:
        p_blue, p_red = predict_proba_names(st.session_state.model,
                                            st.session_state.id2idx,
                                            st.session_state.name2id,
                                            n_blue, n_red)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success("¡Cálculo completado!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("P(Blue gana)", f"{p_blue*100:.1f}%")
    with col2:
        st.metric("P(Red gana)", f"{p_red*100:.1f}%")

# Autocarga si los paths por defecto existen
if "model" not in st.session_state:
    try:
        default_model = os.environ.get("MODEL_PATH", "src/models/checkpoints/bt_transformer_model.pt")
        if os.path.exists(default_model):
            st.session_state.model, st.session_state.id2idx = load_model(default_model, map_location="cpu")
        n2p = os.environ.get("NAME2ID_PATH", "data/artifacts/name2id.json")
        csvg = os.environ.get("CHAMPS_CSV", "data/download_champions.csv")
        if os.path.exists(n2p) or os.path.exists(csvg) or os.path.exists("data/download_champions.csv"):
            st.session_state.name2id = load_name2id(n2p if os.path.exists(n2p) else None,
                                                    csvg if os.path.exists(csvg) else "data/download_champions.csv")
        if "model" in st.session_state and "name2id" in st.session_state:
            st.success("Modelo y mappings cargados automáticamente.")
    except Exception:
        pass
