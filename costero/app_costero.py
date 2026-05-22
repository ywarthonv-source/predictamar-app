# ================================================================
# PredictaMAR Costero v2.1 -- APP PARA CHRISTIAN
# Mapa interactivo Leaflet con todos los puntos
# Disenada para captura de pantalla antes de salir al mar
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
html, body, .stApp { background: #0A1628 !important; font-family: 'Barlow', sans-serif; }
.block-container { padding: 0.75rem !important; max-width: 440px !important; margin: auto !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.hdr { text-align: center; padding: 0.75rem 0 0.5rem; border-bottom: 1px solid #1E3A5F; margin-bottom: 0.75rem; }
.hdr-title { font-family: 'Barlow Condensed', sans-serif; font-size: 1.8rem; font-weight: 900; color: #FFF; letter-spacing: 0.05em; text-transform: uppercase; }
.hdr-sub { font-size: 0.75rem; color: #4A7FA5; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }
.sema-verde { background: linear-gradient(135deg, #0F6E56, #1D9E75); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem; box-shadow: 0 0 20px rgba(29,158,117,0.3); }
.sema-rojo { background: linear-gradient(135deg, #7A1B1B, #C0392B); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem; }
.sema-amarillo { background: linear-gradient(135deg, #7A4A00, #BA7517); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem; }
.sema-text { font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; font-weight: 900; color: white; text-transform: uppercase; letter-spacing: 0.05em; }
.sema-sub { font-size: 0.8rem; color: rgba(255,255,255,0.75); margin-top: 4px; }
.mar-datos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 0.75rem; }
.mar-dato { background: #0F2540; border: 1px solid #1E3A5F; border-radius: 8px; padding: 0.6rem 0.4rem; text-align: center; }
.mar-dato-val { font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem; font-weight: 700; color: #FFF; line-height: 1; }
.mar-dato-lbl { font-size: 0.65rem; color: #4A7FA5; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }
.mar-dato-ok { border-color: #1D9E75; } .mar-dato-warn { border-color: #BA7517; } .mar-dato-bad { border-color: #C0392B; }
.info-bar { background: #0F2540; border: 1px solid #1E3A5F; border-left: 3px solid #185FA5; border-radius: 0 8px 8px 0; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem; font-size: 0.78rem; color: #7AB3D0; line-height: 1.4; }
.sec-title { font-family: 'Barlow Condensed', sans-serif; font-size: 0.75rem; font-weight: 600; color: #4A7FA5; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; padding-bottom: 4px; border-bottom: 1px solid #1E3A5F; }
.sector-item { background: #0F2540; border: 1px solid #1E3A5F; border-radius: 10px; padding: 0.75rem; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
.sector-color { width: 4px; height: 52px; border-radius: 2px; flex-shrink: 0; }
.sector-body { flex: 1; }
.sector-nombre { font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 700; color: #FFF; text-transform: uppercase; letter-spacing: 0.05em; }
.sector-dist { font-size: 0.75rem; color: #4A7FA5; margin-top: 2px; }
.sector-dir { font-size: 0.82rem; color: #7AB3D0; margin-top: 3px; }
.sector-pts { font-size: 0.7rem; color: #2A5A7A; margin-top: 2px; }
.map-btn { display: inline-block; background: #1D9E75; border: none; border-radius: 6px; padding: 8px 12px; color: white; font-size: 0.75rem; font-weight: 600; text-decoration: none; text-align: center; white-space: nowrap; font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.05em; text-transform: uppercase; }
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
    st.markdown('<div style="text-align:center;padding:2rem 0 0.5rem"><span style="font-family:Barlow Condensed,sans-serif;font-size:2.2rem;font-weight:900;color:#FFF;letter-spacing:0.08em;text-transform:uppercase">PredictaMAR</span><br><span style="font-size:0.75rem;color:#4A7FA5;letter-spacing:0.1em;text-transform:uppercase">Puerto Chorrillos</span></div>', unsafe_allow_html=True)
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
    try:
        f = float(v)
        return f if not np.isnan(f) else d
    except: return d

def to_bool(v):
    if isinstance(v, bool): return v
    return str(v).upper() in ("TRUE","1","YES")

def coord_valida(lat, lon):
    return abs(lat) > 1.0 and abs(lon) > 1.0 and lat != 0 and lon != 0

# ================================================================
# HEADER
# ================================================================
st.markdown('<div class="hdr"><div class="hdr-title">PredictaMAR</div><div class="hdr-sub">Puerto Chorrillos</div></div>', unsafe_allow_html=True)

col_act, col_sal = st.columns([3,1])
with col_act:
    if st.button("Actualizar", use_container_width=True):
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
# SEMAFORO
# ================================================================
if mar_adverso:
    st.markdown('<div class="sema-rojo"><div class="sema-text">NO SALIR HOY</div><div class="sema-sub">Oleaje adverso -- condicion peligrosa</div></div>', unsafe_allow_html=True)
elif confianza == "ALTA":
    st.markdown('<div class="sema-verde"><div class="sema-text">BUENAS CONDICIONES</div><div class="sema-sub">Zonas identificadas -- listo para salir</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="sema-amarillo"><div class="sema-text">CONDICIONES MODERADAS</div><div class="sema-sub">Datos limitados -- usar criterio propio</div></div>', unsafe_allow_html=True)

# Metricas
temp_cls = "mar-dato-ok" if 17 <= sst_temp <= 22 else "mar-dato-warn"
surg_cls = "mar-dato-ok" if ind_surg >= 0.70 else "mar-dato-warn"
ola_cls  = "mar-dato-ok" if swh <= 1.0 else "mar-dato-warn" if swh <= 1.5 else "mar-dato-bad"
dirs     = ["N","NE","E","SE","S","SO","O","NO"]
ang      = float(np.degrees(np.arctan2(uo, vo)))
dir_agua = dirs[int((ang + 22.5) / 45) % 8]

st.markdown(f"""
<div class="mar-datos">
  <div class="mar-dato {temp_cls}"><div class="mar-dato-val">{sst_temp:.0f}C</div><div class="mar-dato-lbl">Temperatura</div></div>
  <div class="mar-dato {surg_cls}"><div class="mar-dato-val">{int(ind_surg*100)}%</div><div class="mar-dato-lbl">Surgencia</div></div>
  <div class="mar-dato {ola_cls}"><div class="mar-dato-val">{swh:.1f}m</div><div class="mar-dato-lbl">Oleaje</div></div>
</div>
""", unsafe_allow_html=True)

modo_txt = "Nublado -- modo fisico activo" if not s2_ok else "Satelite optico activo"
st.markdown(f'<div class="info-bar">{modo_txt} -- Agua se mueve hacia el {dir_agua} -- Datos: {hora_str} UTC</div>', unsafe_allow_html=True)

if mar_adverso:
    st.stop()

# ================================================================
# PREPARAR DATOS DE SECTORES
# ================================================================
LAT_CHORRILLOS = -12.157
LON_CHORRILLOS = -77.021

orden_sectores = ["COSTERO", "SUR", "NORTE", "OESTE"]
colores_hex = {"COSTERO": "#1D9E75", "SUR": "#185FA5", "NORTE": "#534AB7", "OESTE": "#BA7517"}
nombres = {"COSTERO": "Orilla (0-3 km)", "SUR": "Sur - Morro Solar", "NORTE": "Norte - Miraflores", "OESTE": "Mar abierto"}
dirs_esp = ["Norte","NorEste","Este","SurEste","Sur","SurOeste","Oeste","NorOeste"]

sectores_data = []
todos_puntos = []  # todos los puntos para el mapa

for sector in orden_sectores:
    df_sec = df_rep[df_rep["sector"] == sector].copy()
    if len(df_sec) == 0:
        continue
    df_sec["rank_num"] = pd.to_numeric(df_sec["rank_sector"], errors="coerce").fillna(99)
    df_sec = df_sec.sort_values("rank_num").reset_index(drop=True)

    puntos_sector = []
    for _, row in df_sec.iterrows():
        lat = to_float(row.get("lat", 0))
        lon = to_float(row.get("lon", 0))
        if not coord_valida(lat, lon):
            continue
        rol  = str(row.get("rol",""))
        rank = int(to_float(row.get("rank_sector", 0)))
        puntos_sector.append({"lat": lat, "lon": lon, "rol": rol, "rank": rank})
        todos_puntos.append({"lat": lat, "lon": lon, "sector": sector, "rank": rank, "rol": rol})

    if not puntos_sector:
        continue

    p1   = puntos_sector[0]
    dist = to_float(df_sec.iloc[0].get("dist_km", 0))
    dlat = p1["lat"] - LAT_CHORRILLOS
    dlon = p1["lon"] - LON_CHORRILLOS
    ang_s = float(np.degrees(np.arctan2(dlon, dlat)))
    dir_s = dirs_esp[int((ang_s + 22.5) / 45) % 8]
    tiempo = int(dist / 15 * 60)

    sectores_data.append({
        "sector": sector,
        "nombre": nombres.get(sector, sector),
        "color":  colores_hex.get(sector, "#4A7FA5"),
        "dist":   dist,
        "dir":    dir_s,
        "tiempo": tiempo,
        "n_puntos": len(puntos_sector),
        "p1_lat": p1["lat"],
        "p1_lon": p1["lon"],
        "puntos": puntos_sector,
    })

# ================================================================
# MAPA INTERACTIVO CON LEAFLET
# Muestra todos los puntos con colores por sector
# Funciona sin API key, permite zoom, pantalla completa
# ================================================================
st.markdown('<div class="sec-title">Mapa -- todos los puntos de hoy</div>', unsafe_allow_html=True)

if todos_puntos:
    # Construir markers para Leaflet
    markers_js = ""
    for p in todos_puntos:
        sector  = p["sector"]
        color   = colores_hex.get(sector, "#4A7FA5")
        rank    = p["rank"]
        # Punto principal mas grande
        radius  = 10 if rank == 1 else 7
        opacity = 1.0 if rank == 1 else 0.75
        label   = "P1" if rank == 1 else f"P{rank}"
        nombre_s = nombres.get(sector, sector)

        markers_js += f"""
L.circleMarker([{p['lat']}, {p['lon']}], {{
    radius: {radius},
    fillColor: '{color}',
    color: 'white',
    weight: 2,
    opacity: 1,
    fillOpacity: {opacity}
}}).bindPopup('<b>{nombre_s}</b><br>{label}<br>{p["lat"]:.4f}, {p["lon"]:.4f}').addTo(map);
"""

    # Pin del muelle
    markers_js += f"""
L.marker([{LAT_CHORRILLOS}, {LON_CHORRILLOS}]).bindPopup('<b>Muelle Chorrillos</b>').addTo(map);
"""

    leaflet_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; }}
  #map {{ height: 100vh; width: 100%; }}
  .leaflet-control-fullscreen {{ display: block; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map', {{zoomControl: true}}).setView([{LAT_CHORRILLOS}, {LON_CHORRILLOS}], 11);

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: 'OpenStreetMap',
    maxZoom: 18
}}).addTo(map);

// Capa satelital disponible para zoom
var satelite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Esri',
    maxZoom: 18
}});

L.control.layers({{"Mapa": L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom:18}}), "Satelite": satelite}}).addTo(map);

{markers_js}

// Leyenda
var legend = L.control({{position: 'bottomleft'}});
legend.onAdd = function(map) {{
    var div = L.DomUtil.create('div', 'info legend');
    div.style.background = 'rgba(10,22,40,0.9)';
    div.style.padding = '8px 10px';
    div.style.borderRadius = '8px';
    div.style.color = 'white';
    div.style.fontSize = '12px';
    div.innerHTML = '<b style="color:#7AB3D0">SECTORES</b><br>' +
        '<span style="color:#1D9E75">&#9679;</span> Orilla<br>' +
        '<span style="color:#185FA5">&#9679;</span> Sur - Morro Solar<br>' +
        '<span style="color:#534AB7">&#9679;</span> Norte - Miraflores<br>' +
        '<span style="color:#BA7517">&#9679;</span> Mar abierto<br>' +
        '<span style="color:#888">P1 = Punto principal</span>';
    return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""

    st.components.v1.html(leaflet_html, height=320, scrolling=False)
    st.caption("Toca un punto para ver detalles. Cambia a 'Satelite' para ver el mar.")

# ================================================================
# LISTA DE SECTORES CON BOTON A GOOGLE MAPS
# ================================================================
st.markdown('<div class="sec-title" style="margin-top:0.75rem">Zonas por sector -- toca para abrir en Google Maps</div>', unsafe_allow_html=True)

for s in sectores_data:
    # URL de Google Maps con todos los puntos del sector marcados
    # Abrir el punto principal con pin
    gmap_url = f"https://www.google.com/maps?q={s['p1_lat']},{s['p1_lon']}&z=14"

    # Para mostrar todos los puntos del sector en Google Maps
    # usar formato de busqueda con waypoints
    if len(s["puntos"]) > 1:
        waypoints = "/".join([f"{p['lat']},{p['lon']}" for p in s["puntos"][:5]])
        gmap_todos = f"https://www.google.com/maps/dir/{s['p1_lat']},{s['p1_lon']}/{waypoints}"
    else:
        gmap_todos = gmap_url

    st.markdown(f"""
<div class="sector-item">
  <div class="sector-color" style="background:{s['color']}"></div>
  <div class="sector-body">
    <div class="sector-nombre">{s['nombre']}</div>
    <div class="sector-dist">{s['dist']:.1f} km -- aprox {s['tiempo']} min navegando</div>
    <div class="sector-dir">Hacia el {s['dir']} -- agua va al {dir_agua}</div>
    <div class="sector-pts">{s['n_puntos']} puntos de prediccion en este sector</div>
  </div>
  <a href="{gmap_url}" target="_blank" class="map-btn">Ver<br>mapa</a>
</div>
""", unsafe_allow_html=True)

# ================================================================
# INSTRUCCIONES MINIMAS
# ================================================================
st.markdown("""
<div class="info-bar" style="margin-top:0.75rem">
  <b>Antes de salir:</b> En el mapa toca cada punto (circulo) para ver las coordenadas.
  Cambia a "Satelite" para ver el mar. Haz captura de pantalla -- no tendras senal en el mar.
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div style="text-align:center;font-size:0.65rem;color:#2A5A7A;margin-top:0.5rem;padding-bottom:0.5rem">PredictaMAR v2.1 -- {datetime.now().strftime("%d/%m/%Y")} -- UNI Startup 2025</div>', unsafe_allow_html=True)
