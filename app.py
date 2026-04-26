import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, datetime
import ephem
import io
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import json

st.set_page_config(
    page_title="PredictaMAR",
    page_icon="🎣",
    layout="centered"
)
# ── Login ─────────────────────────────────────────────────────────
USUARIOS = {
    "randy":    "Luciano1",
    "maik":     "Luciano2",
    "samantha": "Luciano3",
    "usuario4": "Luciano4",
}

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario_actual" not in st.session_state:
    st.session_state["usuario_actual"] = ""

if not st.session_state["autenticado"]:
    st.title("🎣 PredictaMAR")
    st.caption("Sistema de prediccion de zonas de pesca artesanal")
    st.divider()
    st.subheader("Iniciar sesion")
    usuario = st.text_input("Usuario")
    clave   = st.text_input("Contrasena", type="password")
    entrar  = st.button("Entrar", use_container_width=True, type="primary")
    if entrar:
        if usuario in USUARIOS and USUARIOS[usuario] == clave:
            st.session_state["autenticado"]    = True
            st.session_state["usuario_actual"] = usuario
            st.rerun()
        else:
            st.error("Usuario o contrasena incorrectos.")
    st.stop()

st.sidebar.caption("Usuario: " + st.session_state["usuario_actual"])
if st.sidebar.button("Cerrar sesion"):
    st.session_state["autenticado"]    = False
    st.session_state["usuario_actual"] = ""
    st.rerun()
# ── Credenciales Google Drive ─────────────────────────────────────
@st.cache_resource
def conectar_drive():
    creds_info = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)

# ── Cargar cerebro desde Drive ────────────────────────────────────
@st.cache_data(ttl=3600)
def cargar_cerebro():
    try:
        gc = conectar_drive()
        file_id = st.secrets["CEREBRO_FILE_ID"]
        gfile = gc.open_by_key(file_id)

        ws_feat = gfile.worksheet("FEATURES_7D")
        ws_rules = gfile.worksheet("SPECIES_RULES")

        features = pd.DataFrame(ws_feat.get_all_records())
        species_rules = pd.DataFrame(ws_rules.get_all_records())

        return features, species_rules

    except Exception as e:
        st.error("Error cargando cerebro: " + repr(e))
        import traceback
        st.code(traceback.format_exc())
        return None, None

# ── Fase lunar ────────────────────────────────────────────────────
FASES_LUNARES = {
    0: ("🌑", "Luna Nueva"),
    1: ("🌒", "Cuarto Creciente"),
    2: ("🌕", "Luna Llena"),
    3: ("🌘", "Cuarto Menguante"),
}

def get_fase_lunar():
    hoy = ephem.now()
    prev_new = ephem.previous_new_moon(hoy)
    next_new = ephem.next_new_moon(hoy)
    ciclo = next_new - prev_new
    transcurrido = (hoy - prev_new) / ciclo
    if transcurrido < 0.25:
        fase = 0
    elif transcurrido < 0.50:
        fase = 1
    elif transcurrido < 0.75:
        fase = 2
    else:
        fase = 3
    emoji, nombre = FASES_LUNARES[fase]
    return emoji, nombre

# ── Puertos ───────────────────────────────────────────────────────
PUERTOS = {
    "MATARANI":    (-17.00, -72.10),
    "ILO":         (-17.64, -71.34),
    "MORRO_SAMA":  (-17.98, -70.86),
    "TACNA":       (-18.00, -70.50),
    "PUCUSANA":    (-12.48, -76.80),
    "CHORRILLOS":  (-12.18, -77.02),
    "CALLAO":      (-12.06, -77.15),
    "PAITA":       (-5.09,  -81.11),
    "CHIMBOTE":    (-9.08,  -78.59),
    "PISCO":       (-13.70, -76.20),
    "HUACHO":      (-11.10, -77.61),
    "SALAVERRY":   (-8.22,  -78.98),
    "HUANCHACO":   (-8.08,  -79.12),
    "MOLLENDO":    (-16.90, -72.01),
}

HORARIOS = {
    "ANCHOVETA":  "05:30 - 08:00 / 16:30 - 18:30",
    "CHAUCHILLA": "05:30 - 08:00 / 16:30 - 18:30",
    "PEJERREY":   "05:30 - 08:00 / 16:30 - 18:30",
    "BONITO":     "05:00 - 08:00 / 16:00 - 18:30",
    "JUREL":      "05:00 - 08:00 / 16:00 - 18:30",
    "CABALLA":    "05:00 - 08:30 / 16:00 - 18:30",
    "POTA":       "19:00 - 22:00 / 03:00 - 05:00",
    "MERLUZA":    "05:00 - 08:00 / 15:00 - 17:00",
    "LORNA":      "05:30 - 08:00 / 16:30 - 18:30",
    "CABINZA":    "05:30 - 08:00 / 16:30 - 18:30",
}

COLORES = {
    "ANCHOVETA":  ("#0D47A1", "#E3F2FD"),
    "CHAUCHILLA": ("#1565C0", "#E3F2FD"),
    "PEJERREY":   ("#0277BD", "#E1F5FE"),
    "BONITO":     ("#B71C1C", "#FFEBEE"),
    "JUREL":      ("#1B5E20", "#E8F5E9"),
    "CABALLA":    ("#2E7D32", "#E8F5E9"),
    "POTA":       ("#4A148C", "#F3E5F5"),
    "MERLUZA":    ("#E65100", "#FFF3E0"),
    "LORNA":      ("#006064", "#E0F7FA"),
    "CABINZA":    ("#37474F", "#ECEFF1"),
}

EMOJIS = {
    "ANCHOVETA": "🐟", "CHAUCHILLA": "🐟", "PEJERREY": "🐠",
    "BONITO": "🐠",   "JUREL": "🐟",      "CABALLA": "🐟",
    "POTA": "🦑",     "MERLUZA": "🐟",    "LORNA": "🐟",
    "CABINZA": "🐟",
}

# ── Haversine ─────────────────────────────────────────────────────
def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2 = np.radians(np.array(lat2, float))
    lo2 = np.radians(np.array(lon2, float))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ── Scoring ───────────────────────────────────────────────────────
def calcular_score(df, rule):
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["chl_mean_7d", "sst_mean_7d"])
    d["curr_mean_7d"]     = d["curr_mean_7d"].fillna(d["curr_mean_7d"].median())
    d["sal_mean_7d"]      = d["sal_mean_7d"].fillna(d["sal_mean_7d"].median())
    d["front_score_7d"]   = d["front_score_7d"].fillna(0)
    d["chl_cv_7d"]        = d["chl_cv_7d"].fillna(d["chl_cv_7d"].median())
    d["grad_sst_mean_7d"] = d["grad_sst_mean_7d"].fillna(0)
    if d.empty:
        return d

    chl_min  = float(rule.get("chl_min", 0))
    chl_max  = float(rule.get("chl_max", 99))
    sst_min  = float(rule["sst_min_c"])
    sst_max  = float(rule["sst_max_c"])
    curr_max = float(rule["curr_ok_max_ms"])
    sal_min  = float(rule["sal_min"])
    sal_max  = float(rule["sal_max"])

    chl_thr   = max(d["chl_mean_7d"].quantile(float(rule["chl_percentile_high"])), 0.001)
    sc_local  = np.clip(d["chl_mean_7d"] / chl_thr, 0, 2) / 2
    chl_range = max(chl_max - chl_min, 0.01)
    sc_abs    = np.clip((d["chl_mean_7d"] - chl_min) / chl_range, 0, 1)
    d["sc_chl"] = 0.5 * sc_local + 0.5 * sc_abs
    d.loc[d["chl_mean_7d"] > chl_max, "sc_chl"] *= 0.6

    sv = d["sst_mean_7d"].values
    sc = np.ones(len(d))
    sc[sv > sst_max] = np.clip(1 - (sv[sv > sst_max] - sst_max) / 3, 0, 1)
    sc[sv < sst_min] = np.clip(1 - (sst_min - sv[sv < sst_min]) / 3, 0, 1)
    d["sc_sst"]  = sc
    d["sc_grad"] = d["front_score_7d"].fillna(0)

    cv  = d["chl_cv_7d"]
    q80 = max(cv.quantile(0.8), 1e-6)
    d["sc_stab"] = np.clip(1 - cv / q80, 0, 1)

    cv2   = d["curr_mean_7d"].values
    c_lo  = 0.10
    c_hi  = curr_max * 0.60
    sc2   = np.ones(len(d))
    sc2[cv2 < c_lo] = np.clip(0.5 + 0.5 * (cv2[cv2 < c_lo] / c_lo), 0, 1)
    sc2[(cv2 >= c_lo) & (cv2 <= c_hi)] = 1.0
    m_med = (cv2 > c_hi) & (cv2 <= curr_max)
    sc2[m_med] = np.clip(1 - 0.5 * (cv2[m_med] - c_hi) / (curr_max - c_hi), 0.5, 1.0)
    sc2[cv2 > curr_max] = np.clip(1 - (cv2[cv2 > curr_max] - curr_max) / curr_max, 0, 0.5)
    d["sc_curr"] = sc2

    sal_mid   = (sal_min + sal_max) / 2
    sal_range = max((sal_max - sal_min) / 2, 0.01)
    d["sc_sal"] = np.clip(1 - np.abs(d["sal_mean_7d"] - sal_mid) / sal_range, 0, 1)
    d["sc_gchl"] = d["grad_chl_pctl"].fillna(0) if "grad_chl_pctl" in d.columns else 0.5

    w = {
        "chl":  float(rule["w_chl"]),
        "sst":  float(rule["w_sst"]),
        "grad": float(rule["w_grad"]),
        "stab": float(rule["w_stability"]),
        "curr": float(rule["w_curr"]),
        "sal":  float(rule["w_sal"]),
        "gchl": float(rule["w_gchl"]),
    }
    d["score"] = (
        w["chl"]  * d["sc_chl"]  + w["sst"]  * d["sc_sst"]  +
        w["grad"] * d["sc_grad"] + w["stab"] * d["sc_stab"] +
        w["curr"] * d["sc_curr"] + w["sal"]  * d["sc_sal"]  +
        w["gchl"] * d["sc_gchl"]
    )
    tw = sum(w.values())
    if tw > 0:
        d["score"] = d["score"] / tw
    d["score"] = d["score"].clip(0, 1)

    fl = []
    al = []
    for _, r in d.iterrows():
        f = {
            "Chl":  chl_min  <= r["chl_mean_7d"]  <= chl_max,
            "SST":  sst_min  <= r["sst_mean_7d"]  <= sst_max,
            "Corr": r["curr_mean_7d"] <= curr_max,
            "Sal":  sal_min  <= r["sal_mean_7d"]  <= sal_max,
        }
        fl.append(f)
        failed = [k for k, v in f.items() if not v]
        al.append("Alerta: " + ", ".join(failed) + " fuera de rango" if failed else "")
    d["flags"]    = fl
    d["ok_count"] = [sum(f.values()) for f in fl]
    d["alert"]    = al
    return d

def score_a_prob(score, ok):
    if not np.isfinite(score):
        score = 0.0
    base = 30 + score * 50
    adj  = {4: 15, 3: 5, 2: -10, 1: -20, 0: -25}.get(min(ok, 4), -25)
    prob = base + adj
    if ok <= 2:
        prob = min(prob, 49)
    return int(np.clip(prob, 30, 95))

# ── Generar imagen ────────────────────────────────────────────────
def generar_imagen_bytes(idx, row, species, modo, fase_emoji, fase_nombre):
    sp       = species.strip().upper()
    ch, cb   = COLORES.get(sp, ("#0D2B55", "#E8F4FF"))
    em       = EMOJIS.get(sp, "🎣")
    hr       = HORARIOS.get(sp, "05:00-08:00 / 16:00-18:30")
    fl       = row.get("flags", {})
    al       = row.get("alert", "")

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor(cb)
    ax.set_facecolor(cb)

    # Header
    ax.add_patch(FancyBboxPatch((0, 14.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor=ch, edgecolor="none"))
    ax.text(5, 15.55, "PREDICTAMAR  " + em + "  " + sp,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")
    ax.text(5, 15.05, "Zona: " + modo,
            ha="center", va="center", fontsize=8, color="white")

    # Número
    ax.add_patch(plt.Circle((5, 13.8), 0.65, color=ch, zorder=3))
    ax.text(5, 13.8, str(idx + 1),
            ha="center", va="center", fontsize=18,
            fontweight="bold", color="white", zorder=4)

    # Probabilidad
    prob = int(row["prob"])
    pc   = "#1B5E20" if prob >= 70 else "#E65100" if prob >= 50 else "#B71C1C"
    ax.text(5, 12.8, "Probabilidad de pesca: " + str(prob) + "%",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=pc)

    # Alerta
    if al:
        ax.text(5, 12.3, "  " + al,
                ha="center", va="center", fontsize=7.5,
                color="#B71C1C", fontstyle="italic")

    ax.plot([0.5, 9.5], [12.0, 12.0], color=ch, linewidth=1.5, alpha=0.4)

    # Coordenadas
    ax.text(0.5, 11.6, "📍", fontsize=11, va="center")
    ax.text(1.3, 11.6,
            str(abs(row["lat"]))[:7] + "S,  " + str(abs(row["lon"]))[:7] + "W",
            fontsize=10, va="center", color="#212121", fontweight="bold")

    # Variables
    vars_ = [
        ("🌡️", "Temperatura",    str(round(row["sst_mean_7d"],  2)) + " C",   fl.get("SST",  True)),
        ("🧂",  "Salinidad",      str(round(row["sal_mean_7d"],  2)) + " UPS", fl.get("Sal",  True)),
        ("📈",  "Grad. termico",  str(round(row["grad_sst_mean_7d"], 4)) + " C/km", True),
        ("🌿",  "Clorofila-a",    str(round(row["chl_mean_7d"],  4)) + " mg/m3", fl.get("Chl", True)),
        ("🌀",  "Corriente",      str(round(row["curr_mean_7d"], 2)) + " m/s",  fl.get("Corr", True)),
        ("📏",  "Distancia",      str(round(row["dist_km"],      1)) + " km",   True),
        (fase_emoji, "Fase lunar", fase_nombre, True),
    ]

    y = 10.9
    for ev, lb, vl, ok in vars_:
        sc2 = "#1B5E20" if ok else "#B71C1C"
        sym = "OK" if ok else "NO"
        ax.text(0.5, y, ev,  fontsize=9, va="center")
        ax.text(1.3, y, lb,  fontsize=8, va="center", color="#555555")
        ax.text(9.5, y, vl,  fontsize=9, va="center",
                ha="right", fontweight="bold", color="#212121")
        ax.text(9.8, y, sym, fontsize=7, va="center",
                color=sc2, fontweight="bold")
        ax.plot([0.5, 9.5], [y - 0.35, y - 0.35],
                color="#CCCCCC", linewidth=0.5, alpha=0.6)
        y -= 0.72

    # Horario
    ax.add_patch(FancyBboxPatch((0.3, y - 0.25), 9.4, 0.65,
                  boxstyle="round,pad=0.1", facecolor=ch,
                  alpha=0.15, edgecolor=ch, linewidth=1))
    ax.text(0.5, y + 0.08, "⏰", fontsize=9, va="center")
    ax.text(1.3, y + 0.08, "Mejor hora:  " + hr,
            fontsize=8.5, va="center", color=ch, fontweight="bold")
    y -= 0.8

    # Condiciones
    ct = "  ".join([("OK " if v else "NO ") + k for k, v in fl.items()])
    ax.text(5, y, ct, ha="center", va="center", fontsize=8, color="#444444")
    y -= 0.5

    ax.plot([0.5, 9.5], [y + 0.15, y + 0.15], color=ch, linewidth=1, alpha=0.3)
    ax.text(5, y - 0.15,
            "PredictaMAR · Sistema de Corriente de Humboldt · Peru",
            ha="center", va="center", fontsize=6.5,
            color="#888888", style="italic")
    ax.text(5, y - 0.5,
            "Probabilidad operativa estimada",
            ha="center", va="center", fontsize=6,
            color="#AAAAAA", style="italic")

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=cb)
    plt.close()
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════════

st.title("🎣 PredictaMAR")
st.caption("Sistema de prediccion de zonas de pesca artesanal — Corriente de Humboldt, Peru")

# Cargar datos
features, species_rules = cargar_cerebro()

if features is None:
    st.error("No se pudo cargar el cerebro. Verifica la conexion con Drive.")
    st.stop()

st.success("Datos cargados: " + str(len(features)) + " puntos oceanograficos")

st.divider()

# ── Formulario de consulta ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    especie = st.selectbox(
        "🐟 Especie objetivo",
        options=list(species_rules["species"]),
    )

with col2:
    radio_km = st.slider("📏 Radio de busqueda (km)", 20, 200, 80, 10)

modo_busqueda = st.radio(
    "📍 Modo de busqueda",
    ["Por puerto", "Entre dos puertos", "Por coordenadas"],
    horizontal=True,
)

puerto       = None
puerto_desde = None
puerto_hasta = None
lat_input    = None
lon_input    = None

if modo_busqueda == "Por puerto":
    puerto = st.selectbox("Puerto de salida", list(PUERTOS.keys()))

elif modo_busqueda == "Entre dos puertos":
    col3, col4 = st.columns(2)
    with col3:
        puerto_desde = st.selectbox("Puerto origen", list(PUERTOS.keys()))
    with col4:
        puerto_hasta = st.selectbox("Puerto destino", list(PUERTOS.keys()))

else:
    col5, col6 = st.columns(2)
    with col5:
        lat_input = st.number_input("Latitud (negativa)", -22.0, -3.0, -12.0, 0.01)
    with col6:
        lon_input = st.number_input("Longitud (negativa)", -85.0, -68.0, -77.0, 0.01)

top_n = st.selectbox("Numero de puntos recomendados", [1, 2, 3, 4, 5], index=2)

buscar = st.button("🔍 Buscar zonas de pesca", use_container_width=True, type="primary")

# ── Ejecutar búsqueda ─────────────────────────────────────────────
if buscar:
    fase_emoji, fase_nombre = get_fase_lunar()

    # Resolver centro
    if puerto:
        clat, clon = PUERTOS[puerto]
        modo = puerto
    elif puerto_desde and puerto_hasta:
        la1, lo1 = PUERTOS[puerto_desde]
        la2, lo2 = PUERTOS[puerto_hasta]
        clat = (la1 + la2) / 2
        clon = (lo1 + lo2) / 2
        dist = haversine_nm(clat, clon, np.array([la1]), np.array([lo1]))[0]
        radio_nm_auto = dist * 1.2
        modo = "Entre " + puerto_desde + " y " + puerto_hasta
    else:
        clat, clon = lat_input, lon_input
        modo = "(" + str(lat_input) + ", " + str(lon_input) + ")"

    radio_nm = radio_km / 1.852

    # Filtrar por radio
    df = features.copy()
    df["dist_nm"] = haversine_nm(clat, clon, df["lat"].values, df["lon"].values)
    df = df[df["dist_nm"] <= radio_nm].copy()

    if puerto_desde and puerto_hasta:
        df = df[
            (df["lat"] >= min(la1, la2) - 0.5) & (df["lat"] <= max(la1, la2) + 0.5) &
            (df["lon"] >= min(lo1, lo2) - 0.8) & (df["lon"] <= max(lo1, lo2) + 0.8)
        ].copy()

    if df.empty:
        st.warning("Sin puntos en la zona. Amplia el radio.")
        st.stop()

    # Calcular score
    rule = species_rules[species_rules["species"] == especie].iloc[0]
    df   = calcular_score(df, rule)
    df   = df.dropna(subset=["score"])
    df   = df[df["score"] > 0].sort_values("score", ascending=False)

    if df.empty:
        st.warning("Sin puntos con score valido.")
        st.stop()

    res           = df.head(top_n).reset_index(drop=True)
    res["prob"]   = res.apply(lambda r: score_a_prob(r["score"], r["ok_count"]), axis=1)
    res["dist_km"] = (res["dist_nm"] * 1.852).round(1)

    st.divider()
    st.subheader("📊 Resultados — " + especie)
    st.caption("Fase lunar hoy: " + fase_emoji + " " + fase_nombre)

    # Mostrar resultados
    for i, row in res.iterrows():
        with st.expander(
            "Punto " + str(i+1) + " — Prob: " + str(row["prob"]) + "% | " +
            str(round(abs(row["lat"]), 3)) + "S, " +
            str(round(abs(row["lon"]), 3)) + "W | " +
            str(row["dist_km"]) + " km",
            expanded=True
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Temperatura",  str(round(row["sst_mean_7d"], 2))  + " C")
            c2.metric("🌿 Clorofila-a",  str(round(row["chl_mean_7d"], 4))  + " mg/m3")
            c3.metric("🌀 Corriente",    str(round(row["curr_mean_7d"], 2)) + " m/s")
            c4, c5, c6 = st.columns(3)
            c4.metric("🧂 Salinidad",    str(round(row["sal_mean_7d"], 2))  + " UPS")
            c5.metric("📈 Grad. termico",str(round(row["grad_sst_mean_7d"], 4)) + " C/km")
            c6.metric(fase_emoji + " Fase lunar", fase_nombre)
            st.caption("⏰ Mejor hora: " + HORARIOS.get(especie, ""))

            if row.get("alert"):
                st.warning(row["alert"])

            # Generar imagen descargable
            buf = generar_imagen_bytes(i, row, especie, modo, fase_emoji, fase_nombre)
            st.image(buf, use_column_width=True)
            buf.seek(0)
            st.download_button(
                label="⬇️ Descargar imagen punto " + str(i+1),
                data=buf,
                file_name="predictamar_" + especie + "_punto" + str(i+1) + ".png",
                mime="image/png",
                use_container_width=True,
            )

    st.divider()
    st.caption("PredictaMAR v5.0 · Sistema de Corriente de Humboldt · Peru")
