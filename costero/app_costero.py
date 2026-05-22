# ================================================================
# PredictaMAR Costero v2.0 -- APP PARA CHRISTIAN
# Disenada para captura de pantalla antes de salir al mar
# Una sola pantalla, sin scroll, mapa con pins
# Android, sin senial en el mar
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime
import json

st.set_page_config(
    page_title="PredictaMAR",
    page_icon=":fish:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #0A1628 !important;
    font-family: 'Barlow', sans-serif;
}

.block-container {
    padding: 0.75rem !important;
    max-width: 420px !important;
    margin: auto !important;
}

/* Ocultar elementos de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Header */
.hdr {
    text-align: center;
    padding: 1rem 0 0.5rem;
    border-bottom: 1px solid #1E3A5F;
    margin-bottom: 0.75rem;
}
.hdr-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hdr-sub {
    font-size: 0.8rem;
    color: #4A7FA5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}
.hdr-time {
    font-size: 0.7rem;
    color: #2A5A7A;
    margin-top: 4px;
}

/* Semaforo grande */
.sema-verde {
    background: linear-gradient(135deg, #0F6E56, #1D9E75);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 20px rgba(29,158,117,0.3);
}
.sema-rojo {
    background: linear-gradient(135deg, #7A1B1B, #C0392B);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 20px rgba(192,57,43,0.3);
}
.sema-amarillo {
    background: linear-gradient(135deg, #7A4A00, #BA7517);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 20px rgba(186,117,23,0.3);
}
.sema-icon {
    font-size: 2.5rem;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.sema-text {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.sema-sub {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.75);
    margin-top: 4px;
}

/* Metricas del mar */
.mar-datos {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-bottom: 0.75rem;
}
.mar-dato {
    background: #0F2540;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 0.6rem 0.4rem;
    text-align: center;
}
.mar-dato-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
}
.mar-dato-lbl {
    font-size: 0.65rem;
    color: #4A7FA5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}
.mar-dato-ok  { border-color: #1D9E75; }
.mar-dato-warn { border-color: #BA7517; }
.mar-dato-bad  { border-color: #C0392B; }

/* Alerta modo */
.modo-alerta {
    background: #0F2540;
    border: 1px solid #1E3A5F;
    border-left: 3px solid #185FA5;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
    font-size: 0.78rem;
    color: #7AB3D0;
    line-height: 1.4;
}

/* Titulo secciones */
.sec-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #4A7FA5;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1E3A5F;
}

/* Mapa iframe */
.mapa-wrap {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1E3A5F;
    margin-bottom: 0.75rem;
    height: 260px;
}

/* Sectores lista */
.sector-item {
    background: #0F2540;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 0.75rem;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sector-color {
    width: 4px;
    height: 48px;
    border-radius: 2px;
    flex-shrink: 0;
}
.sector-body { flex: 1; }
.sector-nombre {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.sector-dist {
    font-size: 0.75rem;
    color: #4A7FA5;
    margin-top: 2px;
}
.sector-dir {
    font-size: 0.82rem;
    color: #7AB3D0;
    margin-top: 3px;
}
.sector-btn {
    background: #1E3A5F;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    color: #7AB3D0;
    font-size: 0.72rem;
    text-decoration: none;
    white-space: nowrap;
    font-family: 'Barlow', sans-serif;
}

/* Login */
.login-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1.5rem;
}
.login-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
    margin-bottom: 4px;
}
.login-sub {
    font-size: 0.75rem;
    color: #4A7FA5;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOGIN
# ================================================================
USUARIOS = {
    "christian": "chorrillos2025",
    "randy":     "predictamar2025",
    "maik":      "predictamar2025",
    "samantha":  "predictamar2025",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown('<div class="login-logo">PredictaMAR</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Puerto Chorrillos</div>', unsafe_allow_html=True)
    with st.form("login"):
        usuario = st.text_input("", placeholder="Usuario")
        clave   = st.text_input("", type="password", placeholder="Clave")
        btn     = st.form_submit_button("ENTRAR", use_container_width=True)
        if btn:
            if usuario.lower() in USUARIOS and USUARIOS[usuario.lower()] == clave:
                st.session_state.logged_in = True
                st.session_state.usuario   = usuario.lower()
                st.rerun()
            else:
                st.error("Usuario o clave incorrectos")
    st.stop()

# ================================================================
# CARGA DE DATOS
# ================================================================
@st.cache_data(ttl=300)
def cargar_datos():
    try:
        sa_info = json.loads(st.secrets["GOOGLE_SA_JSON"])
        scopes  = ["https://www.googleapis.com/auth/spreadsheets",
                   "https://www.googleapis.com/auth/drive"]
        creds   = SACredentials.from_service_account_info(sa_info, scopes=scopes)
        gc      = gspread.authorize(creds)
        sh      = gc.open_by_key(st.secrets["SHEET_ID"])
        df_rep  = pd.DataFrame(sh.worksheet("costero_reporte").get_all_records())
        return df_rep, None
    except Exception as e:
        return None, str(e)

def to_float(v, d=0.0):
    try: return float(v)
    except: return d

def to_bool(v):
    if isinstance(v, bool): return v
    return str(v).upper() in ("TRUE","1","YES")

# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div class="hdr">
  <div class="hdr-title">PredictaMAR</div>
  <div class="hdr-sub">Puerto Chorrillos</div>
</div>
""", unsafe_allow_html=True)

col_act, col_sal = st.columns([3,1])
with col_act:
    if st.button("Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_sal:
    if st.button("Salir"):
        st.session_state.logged_in = False
        st.rerun()

# ================================================================
# CARGAR DATOS
# ================================================================
df_rep, error = cargar_datos()

if error or df_rep is None or len(df_rep) == 0:
    st.error("Sin datos. Conectate a WiFi y actualiza.")
    st.stop()

fila        = df_rep.iloc[0]
kill_switch = to_bool(fila.get("kill_switch", False))
swh         = to_float(fila.get("swh_medio", 0.8))
sst_temp    = to_float(fila.get("sst_temp_medio", 0))
ind_surg    = to_float(fila.get("indice_surgencia", 0.5))
s2_ok       = to_bool(fila.get("s2_bloom_ok", False))
s1_dias     = int(to_float(fila.get("s1_dias", 99)))
hora_str    = str(fila.get("hora_utc",""))
confianza   = str(fila.get("confianza","MEDIA"))
uo          = to_float(fila.get("uo_medio",0))
vo          = to_float(fila.get("vo_medio",0))

mar_adverso = (swh > 1.5) or kill_switch

# ================================================================
# SEMAFORO GRANDE
# ================================================================
if mar_adverso:
    st.markdown("""
    <div class="sema-rojo">
      <div class="sema-icon">X</div>
      <div class="sema-text">No salir hoy</div>
      <div class="sema-sub">Oleaje adverso -- condicion peligrosa</div>
    </div>
    """, unsafe_allow_html=True)
elif confianza == "ALTA":
    st.markdown("""
    <div class="sema-verde">
      <div class="sema-icon">OK</div>
      <div class="sema-text">Buenas condiciones</div>
      <div class="sema-sub">Zonas identificadas -- listo para salir</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="sema-amarillo">
      <div class="sema-icon">~~</div>
      <div class="sema-text">Condiciones moderadas</div>
      <div class="sema-sub">Datos limitados -- usar criterio propio</div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# METRICAS DEL MAR
# ================================================================
temp_cls  = "mar-dato-ok"  if 17 <= sst_temp <= 22 else "mar-dato-warn"
surg_cls  = "mar-dato-ok"  if ind_surg >= 0.70 else "mar-dato-warn"
ola_cls   = "mar-dato-ok"  if swh <= 1.0 else "mar-dato-warn" if swh <= 1.5 else "mar-dato-bad"

dirs = ["N","NE","E","SE","S","SO","O","NO"]
ang  = float(np.degrees(np.arctan2(uo, vo)))
dir_agua = dirs[int((ang + 22.5) / 45) % 8]

st.markdown(f"""
<div class="mar-datos">
  <div class="mar-dato {temp_cls}">
    <div class="mar-dato-val">{sst_temp:.0f}C</div>
    <div class="mar-dato-lbl">Temperatura</div>
  </div>
  <div class="mar-dato {surg_cls}">
    <div class="mar-dato-val">{int(ind_surg*100)}%</div>
    <div class="mar-dato-lbl">Surgencia</div>
  </div>
  <div class="mar-dato {ola_cls}">
    <div class="mar-dato-val">{swh:.1f}m</div>
    <div class="mar-dato-lbl">Oleaje</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Modo y hora
modo_txt = "Nublado -- modo fisico activo" if not s2_ok else "Satelite optico activo"
st.markdown(f"""
<div class="modo-alerta">
  {modo_txt} -- Agua se mueve hacia el {dir_agua} --
  Actualizado: {hora_str} UTC
</div>
""", unsafe_allow_html=True)

if mar_adverso:
    st.stop()

# ================================================================
# MAPA CON PINS DE GOOGLE MAPS EMBED
# Solo el punto principal de cada sector
# ================================================================
st.markdown('<div class="sec-title">Mapa -- puntos de hoy</div>', unsafe_allow_html=True)

# Obtener punto principal de cada sector
orden_sectores = ["COSTERO", "SUR", "NORTE", "OESTE"]
colores_sector = {
    "COSTERO": "#1D9E75",
    "SUR":     "#185FA5",
    "NORTE":   "#534AB7",
    "OESTE":   "#BA7517"
}
nombres_sector = {
    "COSTERO": "Orilla",
    "SUR":     "Sur - Morro Solar",
    "NORTE":   "Norte - Miraflores",
    "OESTE":   "Mar abierto"
}

puntos_principales = []
for sector in orden_sectores:
    df_sec = df_rep[df_rep["sector"] == sector].copy()
    if len(df_sec) == 0:
        continue
    df_sec["rank_num"] = pd.to_numeric(df_sec["rank_sector"], errors="coerce").fillna(99)
    p1 = df_sec.sort_values("rank_num").iloc[0]
    lat = to_float(p1.get("lat", 0))
    lon = to_float(p1.get("lon", 0))
    if lat != 0 and lon != 0:
        puntos_principales.append({
            "sector": sector,
            "lat": lat,
            "lon": lon,
            "dist": to_float(p1.get("dist_km", 0)),
            "score_local": to_float(p1.get("score_local", 0)),
        })

# Construir URL de Google Maps con multiples pins
# Centro del mapa: Chorrillos
LAT_CHORRILLOS = -12.157
LON_CHORRILLOS = -77.021

if puntos_principales:
    # Mapa embed con markers -- usar iframe de Google Maps
    markers_str = ""
    for p in puntos_principales:
        markers_str += f"markers=color:red%7C{p['lat']},{p['lon']}&"

    # Centro en Chorrillos
    maps_embed = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={LAT_CHORRILLOS},{LON_CHORRILLOS}"
        f"&zoom=11&size=400x250&scale=2"
        f"&maptype=satellite"
        f"&{markers_str}"
        f"key=AIzaSyD-placeholder"
    )

    # Sin API key usar iframe de Google Maps normal
    # Construir URL con todos los puntos para abrir en Google Maps
    query_parts = []
    for i, p in enumerate(puntos_principales):
        query_parts.append(f"{p['lat']},{p['lon']}")

    # Primer punto como destino, resto como waypoints
    if len(query_parts) == 1:
        maps_url = f"https://www.google.com/maps?q={query_parts[0]}"
    else:
        # Abrir mapa centrado en Chorrillos mostrando todos los pins
        maps_url = f"https://www.google.com/maps/search/?api=1&query={LAT_CHORRILLOS},{LON_CHORRILLOS}"

    # Iframe embebido -- mapa de OpenStreetMap sin API key
    # Mostrar el area de Chorrillos con los puntos
    bbox_lat_min = min([p['lat'] for p in puntos_principales]) - 0.05
    bbox_lat_max = max([p['lat'] for p in puntos_principales]) + 0.05
    bbox_lon_min = min([p['lon'] for p in puntos_principales]) - 0.05
    bbox_lon_max = max([p['lon'] for p in puntos_principales]) + 0.05

    osm_url = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={bbox_lon_min}%2C{bbox_lat_min}%2C{bbox_lon_max}%2C{bbox_lat_max}"
        f"&layer=mapnik"
        f"&marker={LAT_CHORRILLOS}%2C{LON_CHORRILLOS}"
    )

    st.markdown(f"""
    <div class="mapa-wrap">
      <iframe src="{osm_url}"
        width="100%" height="260" frameborder="0" scrolling="no"
        style="border:none">
      </iframe>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Toca cada zona abajo para abrir el pin exacto en Google Maps")

# ================================================================
# LISTA DE SECTORES -- SIMPLE Y CLARA
# ================================================================
st.markdown('<div class="sec-title">Zonas recomendadas hoy</div>', unsafe_allow_html=True)

for p in puntos_principales:
    sector   = p["sector"]
    lat      = p["lat"]
    lon      = p["lon"]
    dist     = p["dist"]
    color    = colores_sector.get(sector, "#4A7FA5")
    nombre   = nombres_sector.get(sector, sector)

    # Direccion desde el muelle
    dlat = lat - LAT_CHORRILLOS
    dlon = lon - LON_CHORRILLOS
    ang_sector = float(np.degrees(np.arctan2(dlon, dlat)))
    dirs_esp = ["Norte","NorEste","Este","SurEste","Sur","SurOeste","Oeste","NorOeste"]
    dir_sector = dirs_esp[int((ang_sector + 22.5) / 45) % 8]

    # Link a Google Maps con pin exacto
    gmap_link = f"https://www.google.com/maps?q={lat},{lon}"

    # Tiempo estimado de navegacion (motor 15HP ~ 15 km/h)
    tiempo_min = int(dist / 15 * 60)

    st.markdown(f"""
    <div class="sector-item">
      <div class="sector-color" style="background:{color}"></div>
      <div class="sector-body">
        <div class="sector-nombre">{nombre}</div>
        <div class="sector-dist">{dist:.1f} km del muelle -- aprox {tiempo_min} min navegando</div>
        <div class="sector-dir">Direccion: {dir_sector} -- agua se mueve al {dir_agua}</div>
      </div>
      <a href="{gmap_link}" target="_blank" class="sector-btn">Ver en mapa</a>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# NOTA FINAL
# ================================================================
st.markdown("""
<div class="modo-alerta" style="margin-top:0.75rem;font-size:0.75rem">
  <b>Antes de salir:</b> Toca "Ver en mapa" por cada zona y
  guarda una captura de pantalla. No tendras senial en el mar.
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div style="text-align:center;font-size:0.65rem;color:#2A5A7A;margin-top:0.75rem">PredictaMAR v2.0 -- {datetime.now().strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
