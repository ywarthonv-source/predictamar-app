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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

st.set_page_config(
    page_title="PredictaMAR",
    page_icon="🎣",
    layout="centered"
)

# ── Login ─────────────────────────────────────────────────────────
USUARIOS = {
    "randy":      "Luciano1",
    "maik":       "Luciano2",
    "samantha":   "Luciano3",
    "christian":  "Christian1",
    "pescador1":  "MarPeru1",
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
        return df_rep
    except Exception as e:
        st.error(f"Error cargando reporte: {e}")
        return None

# ── Constantes ────────────────────────────────────────────────────
LAT_CHORRILLOS = -12.15
LON_CHORRILLOS = -77.02

VELOCIDAD_15HP = 8.0   # nudos
VELOCIDAD_40HP = 15.0  # nudos
LATENCIA_S3_H  = 24.0  # horas latencia satelital

COLORES_SEMAFORO = {
    "VERDE":    ("#1B5E20", "#E8F5E9"),
    "AMARILLO": ("#F57F17", "#FFFDE7"),
    "ROJO":     ("#B71C1C", "#FFEBEE"),
}

# ── Funciones ─────────────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1+lat2)/2))
    return round(np.sqrt(dlat**2 + dlon**2), 1)

def tiempo_viaje_horas(dist_km, velocidad_nudos):
    dist_nm = dist_km / 1.852
    return dist_nm / velocidad_nudos

def calcular_coordenada_christian(lat_base, lon_base,
                                   dlat_por_hora, dlon_por_hora,
                                   dist_zona_km, velocidad_nudos):
    t_viaje = tiempo_viaje_horas(dist_zona_km, velocidad_nudos)
    t_total  = LATENCIA_S3_H + t_viaje
    lat_final = lat_base + dlat_por_hora * t_total
    lon_final = lon_base + dlon_por_hora * t_total
    desp_km   = round(np.sqrt((dlat_por_hora*t_total*111)**2 +
                               (dlon_por_hora*t_total*111)**2), 1)
    return round(lat_final, 2), round(lon_final, 2), round(t_total, 1), desp_km

def direccion_cardinal(dlat, dlon):
    if abs(dlat) < 1e-6 and abs(dlon) < 1e-6:
        return "sin desplazamiento"
    angulo = np.degrees(np.arctan2(dlon, dlat))
    dirs   = ["N","NE","E","SE","S","SO","O","NO"]
    idx    = int((angulo + 22.5) / 45) % 8
    return dirs[idx]

def confianza_operacional(score, viirs, era5):
    if score >= 0.65 and viirs and era5:   return "ALTA"
    elif score >= 0.55 and (viirs or era5): return "MEDIA"
    else:                                   return "BAJA"

def contexto_biologico(sst, chl):
    if sst is None or chl is None:
        return "pelágicos costeros"
    sst = float(sst)
    chl = float(chl)
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

# ── Generar tarjeta imagen ────────────────────────────────────────
def generar_tarjeta(zona_nom, punto, idx, conf, ctx_bio, decision, t_total_h):
    score    = float(punto.get("score", 0))
    semaforo = punto.get("semaforo", "ROJO")
    ch, cb   = COLORES_SEMAFORO.get(semaforo, ("#555", "#fff"))

    lat_base = float(punto.get("lat_T16", punto.get("lat_base", 0)))
    lon_base = float(punto.get("lon_T16", punto.get("lon_base", 0)))
    dist     = float(punto.get("dist_km", 0))
    desp_km  = float(punto.get("desp_km", 0))

    # Calcular coordenada final para Christian
    vel      = VELOCIDAD_15HP if "15HP" in zona_nom else VELOCIDAD_40HP
    dlat_h   = float(punto.get("dlat_por_hora", -0.0004))
    dlon_h   = float(punto.get("dlon_por_hora", -0.0004))
    lat_ch, lon_ch, t_tot, desp_total = calcular_coordenada_christian(
        lat_base, lon_base, dlat_h, dlon_h, dist, vel
    )

    dir_txt  = direccion_cardinal(dlat_h, dlon_h)

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.patch.set_facecolor(cb)
    ax.set_facecolor(cb)

    # Header
    ax.add_patch(FancyBboxPatch((0, 18.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor=ch, edgecolor="none"))
    ax.text(5, 19.3, "PREDICTAMAR  🎣  v6.2",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")
    ax.text(5, 18.75, f"Puerto Chorrillos · {datetime.utcnow().strftime('%d %b %Y')}",
            ha="center", va="center", fontsize=8, color="white")

    # Zona y semaforo
    ax.add_patch(plt.Circle((5, 17.5), 0.6, color=ch, zorder=3))
    ax.text(5, 17.5, str(idx+1),
            ha="center", va="center", fontsize=16,
            fontweight="bold", color="white", zorder=4)
    ax.text(5, 16.7, f"{zona_nom}",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=ch)
    ax.text(5, 16.2, f"{semaforo} — Score: {score:.2f}",
            ha="center", va="center", fontsize=9, color=ch)

    ax.plot([0.5, 9.5], [15.8, 15.8], color=ch, linewidth=1, alpha=0.4)

    # Coordenada ahora
    ax.text(0.5, 15.4, "📍", fontsize=10, va="center")
    ax.text(1.3, 15.4, "Posicion satelital ahora:",
            fontsize=8, va="center", color="#555555")
    ax.text(1.3, 15.0, f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W",
            fontsize=10, va="center", color="#212121", fontweight="bold")

    # Coordenada Christian
    ax.text(0.5, 14.4, "🎯", fontsize=10, va="center")
    ax.text(1.3, 14.4, f"Donde ir — llegada en ~{t_tot:.0f}h:",
            fontsize=8, va="center", color="#1B5E20")
    ax.text(1.3, 14.0, f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W",
            fontsize=11, va="center", color="#1B5E20", fontweight="bold")

    # Desplazamiento
    ax.text(0.5, 13.4, "🌊", fontsize=10, va="center")
    ax.text(1.3, 13.4, f"El agua se desplazo {desp_total:.1f} km hacia el {dir_txt}",
            fontsize=8, va="center", color="#0D47A1")

    # Distancia
    ax.text(0.5, 12.9, "📏", fontsize=10, va="center")
    ax.text(1.3, 12.9, f"Distancia desde Chorrillos: {dist:.1f} km",
            fontsize=8, va="center", color="#555555")

    ax.plot([0.5, 9.5], [12.5, 12.5], color=ch, linewidth=1, alpha=0.3)

    # Variables oceanograficas
    sst = punto.get("sst", "—")
    chl = punto.get("chl", "—")
    sc  = punto.get("score", 0)

    vars_ = [
        ("🌡️", "Temp. superficial", f"{float(sst):.1f} C" if sst != "—" else "—"),
        ("🌿", "Clorofila-a",       f"{float(chl):.2f} mg/m3" if chl != "—" else "—"),
        ("📈", "Gradiente oceanico","activo" if sc >= 0.55 else "debil"),
        ("⚓", "Confianza operac.", conf),
    ]

    y = 12.1
    for ev, lb, vl in vars_:
        ax.text(0.5, y, ev,  fontsize=9,  va="center")
        ax.text(1.3, y, lb,  fontsize=8,  va="center", color="#555555")
        ax.text(9.5, y, vl,  fontsize=9,  va="center",
                ha="right", fontweight="bold", color="#212121")
        ax.plot([0.5, 9.5], [y-0.3, y-0.3],
                color="#CCCCCC", linewidth=0.5, alpha=0.5)
        y -= 0.65

    ax.plot([0.5, 9.5], [y+0.1, y+0.1], color=ch, linewidth=1, alpha=0.3)

    # Contexto biologico
    ax.text(0.5, y-0.1, "🐟", fontsize=10, va="center")
    ax.text(1.3, y-0.1, "Compatible con:",
            fontsize=8, va="center", color="#555555")
    ax.text(1.3, y-0.5, ctx_bio,
            fontsize=8, va="center", color="#1B5E20", fontstyle="italic")

    ax.plot([0.5, 9.5], [y-0.9, y-0.9], color=ch, linewidth=1, alpha=0.3)

    # Decision
    dec_color = "#1B5E20" if "RECOMENDADA" in decision else \
                "#E65100" if "EXPLORATORIA" in decision else \
                "#B71C1C"
    ax.add_patch(FancyBboxPatch((0.3, y-1.7), 9.4, 0.7,
                  boxstyle="round,pad=0.1",
                  facecolor=dec_color, alpha=0.15,
                  edgecolor=dec_color, linewidth=1))
    ax.text(5, y-1.35, decision,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=dec_color)

    # Footer
    ax.text(5, 0.4,
            "PredictaMAR · Corriente de Humboldt · Peru",
            ha="center", fontsize=6.5, color="#888888", style="italic")
    ax.text(5, 0.1,
            "Score operacional — no es probabilidad estadistica",
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

# Cargar reporte
df_rep = cargar_reporte()

if df_rep is None or df_rep.empty:
    st.warning("Sin reporte disponible para hoy. El pipeline aun no corrio.")
    st.stop()

# Fecha del reporte
fecha_rep = df_rep["fecha"].iloc[0] if "fecha" in df_rep.columns else "—"
st.info(f"📅 Reporte del: **{fecha_rep}** · {len(df_rep)} zonas analizadas")
st.divider()

# Selector de embarcacion
tipo_emb = st.radio(
    "⛵ Tipo de embarcacion",
    ["🚤 Bote 15HP — Zona A (0-40 km)", "🛥️ Lancha 40HP — Zona B (40-80 km)"],
    horizontal=True
)

zona_key  = "A_15HP" if "15HP" in tipo_emb else "B_40HP"
vel_nudos = VELOCIDAD_15HP if "15HP" in tipo_emb else VELOCIDAD_40HP

st.divider()

# Filtrar zona
df_zona = df_rep[df_rep["zona"] == zona_key].copy()

if df_zona.empty:
    st.warning(f"Sin puntos para zona {zona_key} hoy.")
    st.stop()

# Verificar fuentes
viirs_ok = df_zona["chl_fuente"].str.contains("VIIRS").any() if "chl_fuente" in df_zona.columns else False
era5_ok  = df_zona["ekman_fuente"].str.contains("ERA5").any() if "ekman_fuente" in df_zona.columns else False

# Decision global de zona
df_zona["score"] = pd.to_numeric(df_zona["score"], errors='coerce').fillna(0)
mejor_score  = float(df_zona["score"].max()) if "score" in df_zona.columns else 0
mejor_sem    = df_zona.loc[df_zona["score"].idxmax(), "semaforo"] if "score" in df_zona.columns else "ROJO"
conf_zona    = confianza_operacional(mejor_score, viirs_ok, era5_ok)
dec_zona     = decision_salida(mejor_sem, conf_zona)

ch_zona, cb_zona = COLORES_SEMAFORO.get(mejor_sem, ("#555", "#fff"))

st.markdown(
    f"""
    <div style="background:{cb_zona}; border-left:5px solid {ch_zona};
    padding:14px 18px; border-radius:8px; margin-bottom:16px;">
    <span style="font-size:1.4em; font-weight:bold; color:{ch_zona};">
    {dec_zona}
    </span>
    <br>
    <span style="font-size:0.9em; color:#555;">
    Zona {"A — Botes 15HP" if "A" in zona_key else "B — Lanchas 40HP"} ·
    Score max: {mejor_score:.2f} · Confianza: {conf_zona}
    </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Fuentes activas
col1, col2, col3 = st.columns(3)
col1.metric("🛰️ VIIRS NASA", "✅ Activo" if viirs_ok else "⚠️ Solo CMEMS")
col2.metric("💨 ERA5 Ekman", "✅ Activo" if era5_ok else "⚠️ Proxy")
col3.metric("📊 Puntos zona", len(df_zona))

st.divider()

# Mostrar puntos
st.subheader(f"Zonas recomendadas — {'Botes 15HP' if 'A' in zona_key else 'Lanchas 40HP'}")

for i, (_, punto) in enumerate(df_zona.iterrows()):
    score    = float(punto.get("score", 0))
    semaforo = punto.get("semaforo", "ROJO")
    ch, _    = COLORES_SEMAFORO.get(semaforo, ("#555", "#fff"))

    lat_base = float(punto.get("lat_T16", 0))
    lon_base = float(punto.get("lon_T16", 0))
    dist     = float(punto.get("dist_km", 0))

    # Calcular coordenada para Christian
    dlat_h = float(punto.get("dlat_por_hora", -0.0004))
    dlon_h = float(punto.get("dlon_por_hora", -0.0004))
    lat_ch, lon_ch, t_tot, desp_total = calcular_coordenada_christian(
        lat_base, lon_base, dlat_h, dlon_h, dist, vel_nudos
    )
    dir_txt  = direccion_cardinal(dlat_h, dlon_h)
    conf     = confianza_operacional(score, viirs_ok, era5_ok)
    sst      = punto.get("sst", None)
    chl      = punto.get("chl", None)
    ctx_bio  = contexto_biologico(sst, chl)
    decision = decision_salida(semaforo, conf)

    with st.expander(
        f"Punto {i+1} — {semaforo} | Score {score:.2f} | {dist:.0f} km",
        expanded=(i == 0)
    ):
        # Coordenadas
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📍 Posicion satelital ahora**")
            st.code(f"{abs(lat_base):.2f}S / {abs(lon_base):.2f}W")
        with c2:
            st.markdown(f"**🎯 Donde ir — llegada ~{t_tot:.0f}h**")
            st.code(f"{abs(lat_ch):.2f}S / {abs(lon_ch):.2f}W")

        # Desplazamiento
        st.info(
            f"🌊 El agua se desplazará **{desp_total:.1f} km hacia el {dir_txt}** "
            f"desde la foto satelital hasta tu llegada\n\n"
            f"📏 Distancia desde Chorrillos: **{dist:.1f} km**"
        )

        # Variables
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ SST", f"{float(sst):.1f}°C" if sst else "—")
        m2.metric("🌿 CHL", f"{float(chl):.2f}" if chl else "—")
        m3.metric("⚓ Confianza", conf)
        m4.metric("📊 Score", f"{score:.2f}")

        # Contexto biologico
        st.markdown(f"🐟 **Compatible con:** _{ctx_bio}_")

        # Decision
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

        # Tarjeta descargable
        zona_nom = "Zona A — Botes 15HP" if "A" in zona_key else "Zona B — Lanchas 40HP"
        buf = generar_tarjeta(zona_nom, dict(punto) | {
            "lat_T16": lat_base, "lon_T16": lon_base,
            "dlat_por_hora": dlat_h, "dlon_por_hora": dlon_h,
            "desp_km": desp_total, "sst": sst, "chl": chl
        }, i, conf, ctx_bio, decision, t_tot)

        st.image(buf, use_column_width=True)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Descargar tarjeta punto {i+1}",
            data=buf,
            file_name=f"predictamar_punto{i+1}_{fecha_rep}.png",
            mime="image/png",
            use_container_width=True
        )

st.divider()
st.caption(
    "PredictaMAR v6.2 · Corriente de Humboldt · Peru · "
    "Fuentes: CMEMS + VIIRS NASA + ERA5 + Sentinel-2 · "
    "Score operacional — no es probabilidad estadistica"
)
