# ================================================================
# PredictaMAR v6.2 — APP STREAMLIT
# Puerto Chorrillos — Sistema operacional artesanal
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(
    page_title="PredictaMAR",
    page_icon="🎣",
    layout="centered"
)

# ── Utilidad ──────────────────────────────────────────────────────
def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

# ── Login ─────────────────────────────────────────────────────────
USUARIOS = {
    "randy":     "Luciano1",
    "maik":      "Luciano2",
    "samantha":  "Luciano3",
    "christian": "Christian1",
    "pescador1": "MarPeru1",
}

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario" not in st.session_state:
    st.session_state["usuario"] = ""

if not st.session_state["autenticado"]:
    st.title("🎣 PredictaMAR")
    st.caption("Sistema de prediccion de zonas de pesca · Puerto Chorrillos · Peru")
    st.divider()
    usuario = st.text_input("Usuario")
    clave   = st.text_input("Contrasena", type="password")
    if st.button("Entrar", use_container_width=True, type="primary"):
        if usuario in USUARIOS and USUARIOS[usuario] == clave:
            st.session_state["autenticado"] = True
            st.session_state["usuario"]     = usuario
            st.rerun()
        else:
            st.error("Usuario o contrasena incorrectos.")
    st.stop()

st.sidebar.caption(f"Usuario: {st.session_state['usuario']}")
if st.sidebar.button("Cerrar sesion"):
    st.session_state["autenticado"] = False
    st.rerun()

# ── Conexion Google Sheets ────────────────────────────────────────
@st.cache_resource
def conectar_sheets():
    creds_info = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=1800)
def cargar_reporte():
    try:
        gc     = conectar_sheets()
        sh     = gc.open_by_key(st.secrets["SHEET_ID"])
        ws_rep = sh.worksheet("reporte_diario")
        df_rep = pd.DataFrame(ws_rep.get_all_records())
        for col in ['score','lat_base','lon_base','lat_T8','lon_T8',
                    'lat_T16','lon_T16','lat_T24','lon_T24',
                    'dist_km','desp_km','dlat_por_hora','dlon_por_hora',
                    'sst','chl']:
            if col in df_rep.columns:
                df_rep[col] = pd.to_numeric(df_rep[col], errors='coerce')
        return df_rep
    except Exception as e:
        st.error(f"Error cargando reporte: {e}")
        return None

# ── Constantes ────────────────────────────────────────────────────
LAT_CHORRILLOS = -12.15
LON_CHORRILLOS = -77.02
LATENCIA_S3_H  = 24.0
VELOCIDAD_PROMEDIO = 10.0  # nudos promedio artesanal

COLORES_SEMAFORO = {
    "VERDE":    ("#1B5E20", "#E8F5E9"),
    "AMARILLO": ("#F57F17", "#FFFDE7"),
    "ROJO":     ("#B71C1C", "#FFEBEE"),
}

# ── Funciones ─────────────────────────────────────────────────────
def calcular_coordenada_llegada(lat_base, lon_base,
                                 dlat_h, dlon_h, dist_zona_km):
    dist_nm = dist_zona_km / 1.852
    t_viaje = dist_nm / VELOCIDAD_PROMEDIO
    t_total = LATENCIA_S3_H + t_viaje
    lat_f   = lat_base + dlat_h * t_total
    lon_f   = lon_base + dlon_h * t_total
    desp_km = round(np.sqrt((dlat_h*t_total*111)**2 +
                             (dlon_h*t_total*111)**2), 1)
    return round(lat_f, 2), round(lon_f, 2), round(t_total, 1), desp_km

def direccion_cardinal(dlat, dlon):
    if abs(dlat) < 1e-6 and abs(dlon) < 1e-6:
        return "sin desplazamiento"
    angulo = np.degrees(np.arctan2(dlon, dlat))
    dirs   = ["N","NE","E","SE","S","SO","O","NO"]
    idx    = int((angulo + 22.5) / 45) % 8
    return dirs[idx]

def confianza_operacional(score, viirs, era5):
    if score >= 0.65 and viirs and era5:    return "ALTA"
    elif score >= 0.55 and (viirs or era5): return "MEDIA"
    else:                                   return "BAJA"

def contexto_biologico(sst, chl):
    try:
        sst = float(sst)
        chl = float(chl)
    except:
        return "pelágicos costeros"
    if sst <= 19 and chl >= 2.0:
        return "pelágicos costeros fríos (bonito, jurel, caballa)"
    elif sst <= 22 and chl >= 1.0:
        return "pelágicos costeros (jurel, caballa, perico)"
    elif sst > 22:
        return "pelágicos de aguas cálidas (perico, caballa)"
    else:
        return "pelágicos costeros"

def decision_salida(semaforo, confianza):
    if semaforo == "VERDE" and confianza == "ALTA":
        return "✅ SALIDA RECOMENDADA"
    elif semaforo == "VERDE" or (semaforo == "AMARILLO" and confianza != "BAJA"):
        return "⚡ SALIDA EXPLORATORIA"
    elif semaforo == "AMARILLO" and confianza == "BAJA":
        return "⚠️ SALIDA CON PRECAUCION"
    else:
        return "🚫 NO SALIR HOY"

# ── Generar tarjeta ───────────────────────────────────────────────
def generar_tarjeta(score, semaforo, idx,
                    lat_base, lon_base, lat_ch, lon_ch,
                    desp_total, dir_txt, dist, t_tot,
                    sst, chl, conf, ctx_bio, decision, fecha, radio_km):

    ch, cb = COLORES_SEMAFORO.get(semaforo, ("#555", "#fff"))

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.patch.set_facecolor(cb)
    ax.set_facecolor(cb)

    # Header
    ax.add_patch(FancyBboxPatch((0, 18.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor=ch, edgecolor="none"))
    ax.text(5, 19.3, "PREDICTAMAR    v6.2",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")
    ax.text(5, 18.75, f"Puerto Chorrillos  {fecha}",
            ha="center", va="center", fontsize=8, color="white")

    # Numero
    ax.add_patch(plt.Circle((5, 17.5), 0.6, color=ch, zorder=3))
    ax.text(5, 17.5, str(idx+1),
            ha="center", va="center", fontsize=16,
            fontweight="bold", color="white", zorder=4)
    ax.text(5, 16.7, f"Radio: {radio_km} km desde Chorrillos",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=ch)
    ax.text(5, 16.2, f"{semaforo}  Score: {score:.2f}",
            ha="center", va="center", fontsize=9, color=ch)

    ax.plot([0.5, 9.5], [15.8, 15.8], color=ch, linewidth=1, alpha=0.4)

    # Posicion ahora
    ax.text(0.6, 15.4, "Posicion satelital ahora:",
            fontsize=8, va="center", color="#555555")
    ax.text(0.6, 15.0, f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W",
            fontsize=10, va="center", color="#212121", fontweight="bold")

    # Donde ir
    ax.text(0.6, 14.4, f"Donde ir  llegada ~{t_tot:.0f}h:",
            fontsize=8, va="center", color="#1B5E20")
    ax.text(0.6, 14.0, f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W",
            fontsize=11, va="center", color="#1B5E20", fontweight="bold")

    # Desplazamiento
    ax.text(0.6, 13.4,
            f"Agua se desplazo {desp_total:.1f} km hacia el {dir_txt}",
            fontsize=8, va="center", color="#0D47A1")
    ax.text(0.6, 12.9, f"Distancia desde Chorrillos: {dist:.1f} km",
            fontsize=8, va="center", color="#555555")

    ax.plot([0.5, 9.5], [12.5, 12.5], color=ch, linewidth=1, alpha=0.3)

    # Variables
    vars_ = [
        ("Temp. superficial", f"{safe_float(sst):.1f} C"),
        ("Clorofila-a",       f"{safe_float(chl):.2f} mg/m3"),
        ("Gradiente oceanico","activo" if score >= 0.55 else "debil"),
        ("Confianza operac.", conf),
    ]

    y = 12.1
    for lb, vl in vars_:
        ax.text(0.6, y, lb, fontsize=8, va="center", color="#555555")
        ax.text(9.5, y, vl, fontsize=9, va="center",
                ha="right", fontweight="bold", color="#212121")
        ax.plot([0.5, 9.5], [y-0.3, y-0.3],
                color="#CCCCCC", linewidth=0.5, alpha=0.5)
        y -= 0.65

    ax.plot([0.5, 9.5], [y+0.1, y+0.1], color=ch, linewidth=1, alpha=0.3)

    ax.text(0.6, y-0.1, "Compatible con:", fontsize=8,
            va="center", color="#555555")
    ax.text(0.6, y-0.5, ctx_bio, fontsize=8,
            va="center", color="#1B5E20", fontstyle="italic")

    ax.plot([0.5, 9.5], [y-0.9, y-0.9], color=ch, linewidth=1, alpha=0.3)

    dec_color = "#1B5E20" if "RECOMENDADA" in decision else \
                "#E65100" if "EXPLORATORIA" in decision else "#B71C1C"
    ax.add_patch(FancyBboxPatch((0.3, y-1.7), 9.4, 0.7,
                  boxstyle="round,pad=0.1",
                  facecolor=dec_color, alpha=0.15,
                  edgecolor=dec_color, linewidth=1))
    ax.text(5, y-1.35, decision, ha="center", va="center",
            fontsize=10, fontweight="bold", color=dec_color)

    ax.text(5, 0.4, "PredictaMAR  Corriente de Humboldt  Peru",
            ha="center", fontsize=6.5, color="#888888", style="italic")
    ax.text(5, 0.1, "Score operacional  no es probabilidad estadistica",
            ha="center", fontsize=6, color="#AAAAAA", style="italic")

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=cb)
    plt.close()
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ══════════════════════════════════════════════════════════════════

st.title("🎣 PredictaMAR")
st.caption("Sistema de prediccion de zonas de pesca · Puerto Chorrillos · Peru")

df_rep = cargar_reporte()

if df_rep is None or df_rep.empty:
    st.warning("Sin reporte disponible para hoy. El pipeline aun no corrio.")
    st.stop()

fecha_rep = df_rep["fecha"].iloc[0] if "fecha" in df_rep.columns else "—"
st.info(f"📅 Reporte del: **{fecha_rep}** · {len(df_rep)} zonas analizadas")
st.divider()

# --- Slider radio ---
radio_km = st.slider(
    "📏 Radio de busqueda desde Chorrillos (km)",
    min_value=10,
    max_value=80,
    value=40,
    step=5
)

st.divider()

# Filtrar por radio
df_radio = df_rep[df_rep["dist_km"] <= radio_km].copy()
df_radio["score"] = pd.to_numeric(df_radio["score"], errors='coerce').fillna(0)

if df_radio.empty:
    st.warning(f"Sin puntos dentro de {radio_km} km hoy. Aumenta el radio.")
    st.stop()

# Fuentes
viirs_ok = df_radio["chl_fuente"].str.contains("VIIRS").any() if "chl_fuente" in df_radio.columns else False
era5_ok  = df_radio["ekman_fuente"].str.contains("ERA5").any() if "ekman_fuente" in df_radio.columns else False

# Decision global
mejor_score = float(df_radio["score"].max())
mejor_sem   = df_radio.loc[df_radio["score"].idxmax(), "semaforo"]
conf_zona   = confianza_operacional(mejor_score, viirs_ok, era5_ok)
dec_zona    = decision_salida(mejor_sem, conf_zona)
ch_zona, cb_zona = COLORES_SEMAFORO.get(mejor_sem, ("#555", "#fff"))

st.markdown(
    f"""<div style="background:{cb_zona}; border-left:5px solid {ch_zona};
    padding:14px 18px; border-radius:8px; margin-bottom:16px;">
    <span style="font-size:1.4em; font-weight:bold; color:{ch_zona};">
    {dec_zona}
    </span><br>
    <span style="font-size:0.9em; color:#555;">
    Radio: {radio_km} km ·
    Score max: {mejor_score:.2f} ·
    Confianza: {conf_zona}
    </span></div>""",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
col1.metric("🛰️ VIIRS NASA",  "✅ Activo" if viirs_ok else "⚠️ Solo CMEMS")
col2.metric("💨 ERA5 Ekman",  "✅ Activo" if era5_ok  else "⚠️ Proxy")
col3.metric("📊 Puntos zona", len(df_radio))

st.divider()
st.subheader(f"Zonas recomendadas — Radio {radio_km} km")

# Mostrar top 3
for i, (_, punto) in enumerate(df_radio.nlargest(3, 'score').reset_index(drop=True).iterrows()):
    score    = safe_float(punto.get("score", 0))
    semaforo = str(punto.get("semaforo", "ROJO"))
    lat_base = safe_float(punto.get("lat_T16", punto.get("lat_base", 0)))
    lon_base = safe_float(punto.get("lon_T16", punto.get("lon_base", 0)))
    dist     = safe_float(punto.get("dist_km", 0))
    dlat_h   = safe_float(punto.get("dlat_por_hora", -0.0004))
    dlon_h   = safe_float(punto.get("dlon_por_hora", -0.0004))

    lat_ch, lon_ch, t_tot, desp_total = calcular_coordenada_llegada(
        lat_base, lon_base, dlat_h, dlon_h, dist
    )
    dir_txt  = direccion_cardinal(dlat_h, dlon_h)
    conf     = confianza_operacional(score, viirs_ok, era5_ok)
    sst      = punto.get("sst", None)
    chl      = punto.get("chl", None)
    ctx_bio  = contexto_biologico(sst, chl)
    decision = decision_salida(semaforo, conf)
    fecha    = str(punto.get("fecha", "—"))

    with st.expander(
        f"Punto {i+1} — {semaforo} | Score {score:.2f} | {dist:.0f} km",
        expanded=(i == 0)
    ):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📍 Posicion satelital ahora**")
            st.code(f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W")
        with c2:
            st.markdown(f"**🎯 Donde ir — llegada ~{t_tot:.0f}h**")
            st.code(f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W")

        st.info(
            f"🌊 El agua se desplazara **{desp_total:.1f} km hacia el {dir_txt}** "
            f"desde la foto satelital hasta tu llegada\n\n"
            f"📏 Distancia desde Chorrillos: **{dist:.1f} km**"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ SST",       f"{safe_float(sst):.1f}C"  if sst else "—")
        m2.metric("🌿 CHL",       f"{safe_float(chl):.2f}"   if chl else "—")
        m3.metric("⚓ Confianza", conf)
        m4.metric("📊 Score",     f"{score:.2f}")

        st.markdown(f"🐟 **Compatible con:** _{ctx_bio}_")

        dec_color = "#1B5E20" if "RECOMENDADA" in decision else \
                    "#E65100" if "EXPLORATORIA" in decision else "#B71C1C"
        st.markdown(
            f"""<div style="border:2px solid {dec_color}; border-radius:8px;
            padding:10px; text-align:center; margin:8px 0;">
            <span style="font-size:1.2em; font-weight:bold; color:{dec_color};">
            {decision}
            </span></div>""",
            unsafe_allow_html=True
        )

        buf = generar_tarjeta(
            score, semaforo, i,
            lat_base, lon_base, lat_ch, lon_ch,
            desp_total, dir_txt, dist, t_tot,
            sst, chl, conf, ctx_bio, decision, fecha, radio_km
        )
        st.image(buf, use_column_width=True)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Descargar tarjeta punto {i+1}",
            data=buf,
            file_name=f"predictamar_punto{i+1}_{fecha}.png",
            mime="image/png",
            use_container_width=True
        )

st.divider()
st.caption(
    "PredictaMAR v6.2 · Corriente de Humboldt · Peru · "
    "Fuentes: CMEMS + VIIRS NASA + ERA5 + Sentinel-2 · "
    "Score operacional — no es probabilidad estadistica"
)
