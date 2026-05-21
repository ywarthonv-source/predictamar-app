# ================================================================
# PredictaMAR v6.2 — APP STREAMLIT
# Puerto Chorrillos — Sistema operacional artesanal
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import io
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(
    page_title="PredictaMAR",
    page_icon="🎣",
    layout="centered"
)

def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def score_a_indice(score):
    base = 30 + safe_float(score) * 50
    return int(np.clip(base, 30, 95))

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

# ── Sheets ────────────────────────────────────────────────────────
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
                    'sst','chl','s2_cobertura']:
            if col in df_rep.columns:
                df_rep[col] = pd.to_numeric(df_rep[col], errors='coerce')
        return df_rep
    except Exception as e:
        st.error(f"Error cargando reporte: {e}")
        return None

@st.cache_data(ttl=1800)
def cargar_ipo():
    try:
        gc     = conectar_sheets()
        sh     = gc.open_by_key(st.secrets["SHEET_ID"])
        ws_ipo = sh.worksheet("ipo_zonas")
        df_ipo = pd.DataFrame(ws_ipo.get_all_records())
        for col in ['ipo','lat_base','lon_base','n_corridas']:
            if col in df_ipo.columns:
                df_ipo[col] = pd.to_numeric(df_ipo[col], errors='coerce')
        return df_ipo
    except:
        return pd.DataFrame()

# ── Constantes ────────────────────────────────────────────────────
LATENCIA_S3_H      = 24.0
VELOCIDAD_PROMEDIO = 10.0

COLORES_SEMAFORO = {
    "VERDE":    ("#1B5E20", "#E8F5E9"),
    "AMARILLO": ("#F57F17", "#FFFDE7"),
    "ROJO":     ("#B71C1C", "#FFEBEE"),
}

HORAS_SALIDA = {
    "Ahora mismo (0-2 horas)":       1,
    "Esta tarde (2-6 horas)":        4,
    "Esta noche (6-12 horas)":       9,
    "Manana temprano (12-24 horas)": 18,
    "Manana tarde (24-36 horas)":    30,
}

# ── Funciones ─────────────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1+lat2)/2))
    return np.sqrt(dlat**2 + dlon**2)

def calcular_coordenada_llegada(lat_base, lon_base,
                                 dlat_h, dlon_h,
                                 dist_zona_km, horas_salida):
    dist_nm  = dist_zona_km / 1.852
    t_viaje  = dist_nm / VELOCIDAD_PROMEDIO
    t_total  = LATENCIA_S3_H + horas_salida + t_viaje
    lat_f    = lat_base + dlat_h * t_total
    lon_f    = lon_base + dlon_h * t_total
    desp_km  = round(np.sqrt((dlat_h*t_total*111)**2 +
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
        return "pelagicos costeros"
    if sst <= 19 and chl >= 2.0:
        return "pelagicos costeros frios (bonito, jurel, caballa)"
    elif sst <= 22 and chl >= 1.0:
        return "pelagicos costeros (jurel, caballa, perico)"
    elif sst > 22:
        return "pelagicos de aguas calidas (perico, caballa)"
    else:
        return "pelagicos costeros"

def decision_salida(semaforo, confianza):
    if semaforo == "VERDE" and confianza == "ALTA":
        return "SALIDA RECOMENDADA"
    elif semaforo == "VERDE" or (semaforo == "AMARILLO" and confianza != "BAJA"):
        return "SALIDA EXPLORATORIA"
    elif semaforo == "AMARILLO" and confianza == "BAJA":
        return "SALIDA CON PRECAUCION"
    else:
        return "NO SALIR HOY"

def decision_emoji(decision):
    if "RECOMENDADA"  in decision: return "✅"
    if "EXPLORATORIA" in decision: return "⚡"
    if "PRECAUCION"   in decision: return "⚠️"
    return "🚫"

def buscar_ipo(lat, lon, df_ipo, radio_km=15):
    if df_ipo is None or len(df_ipo) == 0:
        return None, None, None
    df_ipo = df_ipo.copy()
    df_ipo['dist'] = df_ipo.apply(
        lambda r: distancia_km(lat, lon,
                               safe_float(r['lat_base']),
                               safe_float(r['lon_base'])), axis=1
    )
    cercanos = df_ipo[df_ipo['dist'] <= radio_km]
    if len(cercanos) == 0:
        return None, None, None
    mejor = cercanos.loc[cercanos['ipo'].idxmax()]
    return safe_float(mejor['ipo']), str(mejor['ipo_label']), int(mejor['n_corridas'])

def render_ipo(ipo_val, ipo_label, n_corridas):
    if ipo_val is None:
        return
    max_corridas = 4
    n = min(int(n_corridas), max_corridas)
    barras_llenas = round(ipo_val * max_corridas)
    barra = "█" * barras_llenas + "░" * (max_corridas - barras_llenas)

    if ipo_label == "CONFIRMADA":
        color = "#1B5E20"
        emoji = "🟢"
    elif ipo_label == "EN OBSERVACION":
        color = "#F57F17"
        emoji = "🟡"
    else:
        color = "#B71C1C"
        emoji = "🔴"

    st.markdown(
        f"""<div style="background:#f8f8f8; border-left:4px solid {color};
        padding:8px 12px; border-radius:6px; margin:8px 0;">
        <span style="font-size:0.9em; color:{color}; font-weight:bold;">
        {emoji} Persistencia: {barra} {n}/{max_corridas} corridas — {ipo_label}
        </span>
        </div>""",
        unsafe_allow_html=True
    )

# ── Tarjeta ───────────────────────────────────────────────────────
def generar_tarjeta(score, semaforo, idx,
                    lat_base, lon_base, lat_ch, lon_ch,
                    desp_total, dir_txt, dist, t_tot,
                    sst, chl, conf, ctx_bio, decision,
                    fecha, radio_km, hora_salida_txt,
                    ipo_val, ipo_label, n_corridas):

    ch, cb   = COLORES_SEMAFORO.get(semaforo, ("#555", "#fff"))
    indice   = score_a_indice(score)
    ic_color = "#1B5E20" if indice >= 70 else \
               "#E65100" if indice >= 55 else "#B71C1C"

    fig, ax = plt.subplots(figsize=(5, 11.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 23)
    ax.axis("off")
    fig.patch.set_facecolor(cb)
    ax.set_facecolor(cb)

    # Header
    ax.add_patch(FancyBboxPatch((0, 21.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor=ch, edgecolor="none"))
    ax.text(5, 22.3, "PREDICTAMAR   v6.2",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")
    ax.text(5, 21.75, f"Puerto Chorrillos   {fecha}",
            ha="center", va="center", fontsize=8, color="white")

    # Numero
    ax.add_patch(plt.Circle((5, 20.5), 0.65, color=ch, zorder=3))
    ax.text(5, 20.5, str(idx+1),
            ha="center", va="center", fontsize=16,
            fontweight="bold", color="white", zorder=4)

    # Indice
    ax.text(5, 19.5, f"Indice operativo: {indice}%",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=ic_color)
    ax.text(5, 19.0, f"{semaforo}   Radio: {radio_km} km",
            ha="center", va="center", fontsize=9, color=ch)

    # IPO
    if ipo_val is not None:
        max_c = 4
        barras = round(ipo_val * max_c)
        barra  = "█" * barras + "░" * (max_c - barras)
        ipo_color = "#1B5E20" if ipo_label == "CONFIRMADA" else \
                    "#F57F17" if ipo_label == "EN OBSERVACION" else "#B71C1C"
        ax.text(5, 18.5, f"Persistencia: {barra} {n_corridas}/4 — {ipo_label}",
                ha="center", va="center", fontsize=8,
                color=ipo_color, fontweight="bold")

    ax.plot([0.5, 9.5], [18.1, 18.1], color=ch, linewidth=1.5, alpha=0.4)

    # Hora salida
    ax.text(0.6, 17.7, f"Salida: {hora_salida_txt}",
            fontsize=8, va="center", color="#0D47A1", fontstyle="italic")

    ax.plot([0.5, 9.5], [17.4, 17.4], color=ch, linewidth=0.5, alpha=0.3)

    # Coordenadas
    ax.text(0.6, 17.0, "Llega aqui primero:",
            fontsize=8, va="center", color="#555555")
    ax.text(0.6, 16.6, f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W",
            fontsize=10, va="center", color="#212121", fontweight="bold")

    ax.text(0.6, 16.0,
            f"Si no hay actividad en 30 min, avanza {desp_total:.1f} km al {dir_txt}:",
            fontsize=7.5, va="center", color="#1B5E20")
    ax.text(0.6, 15.6, f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W",
            fontsize=10, va="center", color="#1B5E20", fontweight="bold")

    ax.text(0.6, 15.0,
            f"Distancia desde Chorrillos: {dist:.1f} km",
            fontsize=8, va="center", color="#555555")

    ax.plot([0.5, 9.5], [14.6, 14.6], color=ch, linewidth=1, alpha=0.3)

    # Variables
    vars_ = [
        ("Temp. superficial", f"{safe_float(sst):.1f} C"),
        ("Clorofila-a",       f"{safe_float(chl):.2f} mg/m3"),
        ("Gradiente oceanico","activo" if score >= 0.55 else "debil"),
        ("Confianza operac.", conf),
    ]

    y = 14.2
    for lb, vl in vars_:
        ax.text(0.6, y, lb, fontsize=8, va="center", color="#555555")
        ax.text(9.5, y, vl, fontsize=9, va="center",
                ha="right", fontweight="bold", color="#212121")
        ax.plot([0.5, 9.5], [y-0.3, y-0.3],
                color="#CCCCCC", linewidth=0.5, alpha=0.5)
        y -= 0.65

    ax.plot([0.5, 9.5], [y+0.1, y+0.1], color=ch, linewidth=1, alpha=0.3)

    ax.text(0.6, y-0.15, "Compatible con:",
            fontsize=8, va="center", color="#555555")
    ax.text(0.6, y-0.55, ctx_bio,
            fontsize=8, va="center", color="#1B5E20", fontstyle="italic")

    ax.plot([0.5, 9.5], [y-0.95, y-0.95], color=ch, linewidth=1, alpha=0.3)

    dec_color = "#1B5E20" if "RECOMENDADA" in decision else \
                "#E65100" if "EXPLORATORIA" in decision else "#B71C1C"
    ax.add_patch(FancyBboxPatch((0.3, y-1.85), 9.4, 0.75,
                  boxstyle="round,pad=0.1",
                  facecolor=dec_color, alpha=0.15,
                  edgecolor=dec_color, linewidth=1))
    ax.text(5, y-1.5, decision,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=dec_color)

    ax.text(5, 0.4, "PredictaMAR   Corriente de Humboldt   Peru",
            ha="center", fontsize=6.5, color="#888888", style="italic")
    ax.text(5, 0.1, "Indice operativo   no es probabilidad estadistica",
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
df_ipo = cargar_ipo()

if df_rep is None or df_rep.empty:
    st.warning("Sin reporte disponible para hoy. El pipeline aun no corrio.")
    st.stop()

fecha_rep = df_rep["fecha"].iloc[0] if "fecha" in df_rep.columns else "—"
st.info(f"📅 Reporte del: **{fecha_rep}** · {len(df_rep)} puntos disponibles")
st.divider()

# Slider radio
radio_km = st.slider(
    "📏 Radio de busqueda desde Chorrillos (km)",
    min_value=10, max_value=80, value=40, step=5
)

# Selector hora salida
st.markdown("**⏰ Cuando piensas salir?**")
hora_salida_sel = st.radio(
    "", list(HORAS_SALIDA.keys()), index=3, horizontal=False
)
horas_hasta_salida = HORAS_SALIDA[hora_salida_sel]

st.divider()

# Filtrar por radio
df_radio = df_rep[df_rep["dist_km"] <= radio_km].copy()
df_radio["score"] = pd.to_numeric(df_radio["score"], errors='coerce').fillna(0)

if df_radio.empty:
    # No hay puntos en el radio — mostrar los 3 más cercanos disponibles
    df_radio = df_rep.copy()
    df_radio["score"] = pd.to_numeric(df_radio["score"], errors='coerce').fillna(0)
    mejor_cercano = df_radio.nsmallest(1, "dist_km")["dist_km"].values[0]
    st.warning(
        f"⚠️ Sin puntos dentro de {radio_km} km hoy. "
        f"Las mejores condiciones están a {mejor_cercano:.0f} km de Chorrillos. "
        f"Mostrando los puntos más cercanos disponibles."
    )

# Fuentes
viirs_ok = df_radio["chl_fuente"].str.contains("VIIRS").any() \
           if "chl_fuente" in df_radio.columns else False
era5_ok  = df_radio["ekman_fuente"].str.contains("ERA5").any() \
           if "ekman_fuente" in df_radio.columns else False
s2_cob   = df_radio["s2_cobertura"].iloc[0] \
           if "s2_cobertura" in df_radio.columns else 0

# Decision global
mejor_score  = float(df_radio["score"].max())
mejor_sem    = df_radio.loc[df_radio["score"].idxmax(), "semaforo"]
conf_zona    = confianza_operacional(mejor_score, viirs_ok, era5_ok)
dec_zona     = decision_salida(mejor_sem, conf_zona)
dec_emoji_g  = decision_emoji(dec_zona)
indice_zona  = score_a_indice(mejor_score)
ch_zona, cb_zona = COLORES_SEMAFORO.get(mejor_sem, ("#555", "#fff"))

st.markdown(
    f"""<div style="background:{cb_zona}; border-left:5px solid {ch_zona};
    padding:14px 18px; border-radius:8px; margin-bottom:16px;">
    <span style="font-size:1.4em; font-weight:bold; color:{ch_zona};">
    {dec_emoji_g} {dec_zona}
    </span><br>
    <span style="font-size:0.9em; color:#555;">
    Radio: {radio_km} km ·
    Indice max: {indice_zona}% ·
    Confianza: {conf_zona} ·
    Salida: {hora_salida_sel.split("(")[0].strip()}
    </span></div>""",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
col1.metric("🛰️ VIIRS NASA",   "✅ Activo" if viirs_ok else "⚠️ Solo CMEMS")
col2.metric("💨 ERA5 Ekman",   "✅ Activo" if era5_ok  else "⚠️ Proxy")
col3.metric("🛰️ S2 cobertura", f"{s2_cob}%")

st.divider()
st.subheader(f"Top 3 zonas — Radio {radio_km} km")

# Top 3
for i, (_, punto) in enumerate(
    df_radio.nlargest(3, 'score').reset_index(drop=True).iterrows()
):
    score    = safe_float(punto.get("score", 0))
    semaforo = str(punto.get("semaforo", "ROJO"))
    indice   = score_a_indice(score)
    lat_base = safe_float(punto.get("lat_T16", punto.get("lat_base", 0)))
    lon_base = safe_float(punto.get("lon_T16", punto.get("lon_base", 0)))
    dist     = safe_float(punto.get("dist_km", 0))
    dlat_h   = safe_float(punto.get("dlat_por_hora", -0.0004))
    dlon_h   = safe_float(punto.get("dlon_por_hora", -0.0004))

    lat_ch, lon_ch, t_tot, desp_total = calcular_coordenada_llegada(
        lat_base, lon_base, dlat_h, dlon_h, dist, horas_hasta_salida
    )
    dir_txt  = direccion_cardinal(dlat_h, dlon_h)
    conf     = confianza_operacional(score, viirs_ok, era5_ok)
    sst      = punto.get("sst", None)
    chl      = punto.get("chl", None)
    ctx_bio  = contexto_biologico(sst, chl)
    decision = decision_salida(semaforo, conf)
    dec_em   = decision_emoji(decision)
    fecha    = str(punto.get("fecha", "—"))

    # IPO
    ipo_val, ipo_label, n_corridas = buscar_ipo(
        safe_float(punto.get("lat_base", lat_base)),
        safe_float(punto.get("lon_base", lon_base)),
        df_ipo
    )

    with st.expander(
        f"Punto {i+1} — {semaforo} | Indice: {indice}% | {dist:.0f} km",
        expanded=(i == 0)
    ):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📍 Llega aqui primero**")
            st.code(f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W")
        with c2:
            st.markdown(f"**🎯 Si no hay actividad, avanza al {dir_txt}**")
            st.code(f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W")

        st.info(
            f"🌊 Si no encuentras actividad en 30 min, avanza "
            f"**{desp_total:.1f} km hacia el {dir_txt}**\n\n"
            f"📏 Distancia desde Chorrillos: **{dist:.1f} km** · "
            f"Salida: **{hora_salida_sel.split('(')[0].strip()}**"
        )

        # IPO visible
        render_ipo(ipo_val, ipo_label, n_corridas)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ SST",              f"{safe_float(sst):.1f}C" if sst else "—")
        m2.metric("🌿 CHL",              f"{safe_float(chl):.2f}"  if chl else "—")
        m3.metric("⚓ Confianza",        conf)
        m4.metric("📊 Indice operativo", f"{indice}%")

        st.markdown(f"🐟 **Compatible con:** _{ctx_bio}_")

        dec_color = "#1B5E20" if "RECOMENDADA" in decision else \
                    "#E65100" if "EXPLORATORIA" in decision else "#B71C1C"
        st.markdown(
            f"""<div style="border:2px solid {dec_color}; border-radius:8px;
            padding:10px; text-align:center; margin:8px 0;">
            <span style="font-size:1.2em; font-weight:bold; color:{dec_color};">
            {dec_em} {decision}
            </span></div>""",
            unsafe_allow_html=True
        )

        buf = generar_tarjeta(
            score, semaforo, i,
            lat_base, lon_base, lat_ch, lon_ch,
            desp_total, dir_txt, dist, t_tot,
            sst, chl, conf, ctx_bio, decision,
            fecha, radio_km, hora_salida_sel.split("(")[0].strip(),
            ipo_val, ipo_label, n_corridas if n_corridas else 1
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
    "Indice operativo — no es probabilidad estadistica"
)
