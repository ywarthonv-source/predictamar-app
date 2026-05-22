# ================================================================
# PredictaMAR Costero v1.2 -- APP STREAMLIT PARA CHRISTIAN
# Interfaz simple para pescador artesanal de Chorrillos
# Datos desde Google Sheet cerebro_7d -> costero_reporte
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime
import json
import os

# -- Configuracion de pagina
st.set_page_config(
    page_title="PredictaMAR Costero",
    page_icon=":fish:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -- Estilos CSS minimalistas para celular
st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .block-container { padding: 0.5rem 1rem; max-width: 480px; margin: auto; }
    h1 { font-size: 1.4rem !important; margin-bottom: 0.25rem; }
    h2 { font-size: 1.1rem !important; margin-bottom: 0.25rem; }
    h3 { font-size: 1rem !important; margin-bottom: 0.25rem; }
    .semaforo-verde    { background:#1D9E75; color:white; padding:8px 16px; border-radius:8px; font-weight:bold; font-size:1.1rem; text-align:center; margin:4px 0; }
    .semaforo-amarillo { background:#BA7517; color:white; padding:8px 16px; border-radius:8px; font-weight:bold; font-size:1.1rem; text-align:center; margin:4px 0; }
    .semaforo-rojo     { background:#A32D2D; color:white; padding:8px 16px; border-radius:8px; font-weight:bold; font-size:1.1rem; text-align:center; margin:4px 0; }
    .semaforo-adverso  { background:#2C2C2A; color:white; padding:8px 16px; border-radius:8px; font-weight:bold; font-size:1.1rem; text-align:center; margin:4px 0; }
    .sector-header     { background:#0C447C; color:white; padding:8px 12px; border-radius:8px; font-weight:bold; margin:8px 0 4px; }
    .zona-card         { border:1px solid #D3D1C7; border-radius:8px; padding:8px 12px; margin:4px 0; background:#F8F8F6; }
    .zona-nucleo       { border-left:4px solid #1D9E75; }
    .zona-variante     { border-left:4px solid #185FA5; }
    .zona-trampa       { border-left:4px solid #888780; }
    .dato-ok   { color:#1D9E75; font-weight:bold; }
    .dato-warn { color:#BA7517; font-weight:bold; }
    .dato-no   { color:#A32D2D; font-weight:bold; }
    .confianza-alta  { background:#E1F5EE; color:#085041; padding:4px 10px; border-radius:4px; font-size:0.85rem; }
    .confianza-media { background:#FAEEDA; color:#633806; padding:4px 10px; border-radius:4px; font-size:0.85rem; }
    .confianza-baja  { background:#FAECE7; color:#4A1B0C; padding:4px 10px; border-radius:4px; font-size:0.85rem; }
    .footer { font-size:0.75rem; color:#888780; text-align:center; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# CONEXION A GOOGLE SHEETS
# ================================================================
@st.cache_data(ttl=300)  # cache 5 minutos
def cargar_datos():
    try:
        # Credenciales desde secrets de Streamlit
        sa_info = json.loads(st.secrets["GOOGLE_SA_JSON"])
        scopes  = ["https://www.googleapis.com/auth/spreadsheets",
                   "https://www.googleapis.com/auth/drive"]
        creds   = SACredentials.from_service_account_info(sa_info, scopes=scopes)
        gc      = gspread.authorize(creds)
        sh      = gc.open_by_key(st.secrets["SHEET_ID"])

        df_rep  = pd.DataFrame(sh.worksheet('costero_reporte').get_all_records())
        df_ipo  = pd.DataFrame(sh.worksheet('costero_ipo').get_all_records())

        return df_rep, df_ipo, None
    except Exception as e:
        return None, None, str(e)

# ================================================================
# HELPERS
# ================================================================
def semaforo_html(s):
    s = str(s).upper()
    if "ADVERSO" in s:
        return '<div class="semaforo-adverso">ADVERSO -- NO SALIR</div>'
    if "VERDE" in s:
        return '<div class="semaforo-verde">VERDE LOCAL</div>'
    if "AMARILLO" in s:
        return '<div class="semaforo-amarillo">AMARILLO LOCAL</div>'
    return '<div class="semaforo-rojo">CONDICION BAJA</div>'

def confianza_html(c):
    c = str(c).upper()
    if c == "ALTA":
        return '<span class="confianza-alta">Confianza ALTA</span>'
    if c == "MEDIA":
        return '<span class="confianza-media">Confianza MEDIA</span>'
    return '<span class="confianza-baja">Confianza BAJA</span>'

def rol_label(rol):
    rol = str(rol).upper()
    if "NUCLEO" in rol:   return "[1] Punto principal"
    if "VARIANTE_1" in rol: return "[2] Alternativa A"
    if "VARIANTE_2" in rol: return "[3] Alternativa B"
    if "TRAMPA_1" in rol: return "[4] Refugio 1"
    if "TRAMPA_2" in rol: return "[5] Refugio 2"
    return rol

def sector_nombre(s):
    nombres = {
        "COSTERO": "Orilla (0-3 km)",
        "SUR":     "Sur -- Morro Solar",
        "NORTE":   "Norte -- Miraflores",
        "OESTE":   "Oeste -- Mar abierto"
    }
    return nombres.get(str(s).upper(), s)

def coordenadas_link(lat, lon):
    return f"https://www.google.com/mapsq={lat},{lon}"

# ================================================================
# INTERFAZ PRINCIPAL
# ================================================================
st.markdown("#  PredictaMAR Costero")
st.markdown("**Puerto Chorrillos  Lima**")

# Boton de actualizar
col1, col2 = st.columns([3,1])
with col2:
    if st.button(" Actualizar"):
        st.cache_data.clear()
        st.rerun()

# Cargar datos
df_rep, df_ipo, error = cargar_datos()

if error:
    st.error(f"Error al cargar datos: {error}")
    st.stop()

if df_rep is None or len(df_rep) == 0:
    st.warning("Sin datos disponibles. El pipeline corre a las 3AM y 3PM Lima.")
    st.stop()

# ================================================================
# ESTADO DEL MAR HOY
# ================================================================
fila = df_rep.iloc[0]
fecha_str   = str(fila.get('fecha', ''))
hora_str    = str(fila.get('hora_utc', ''))
confianza   = str(fila.get('confianza', 'MEDIA'))
kill_switch = bool(fila.get('kill_switch', False))
swh         = float(fila.get('swh_medio', 0.8))
sst_temp    = fila.get('sst_temp_medio', None)
ind_surg    = float(fila.get('indice_surgencia', 0.5))
s2_ok       = bool(fila.get('s2_bloom_ok', False))
sst_ok      = bool(fila.get('sst_ok', False))
era5_ok     = bool(fila.get('era5_ok', False))
s1_ok       = fila.get('s1_dias', 99)
s1_dias     = int(s1_ok) if str(s1_ok).lstrip('-').isdigit() else 99

st.markdown("---")
st.markdown("### Estado del mar")

# Semaforo general
if kill_switch or swh > 1.5:
    st.markdown('<div class="semaforo-adverso">ADVERSO -- NO SALIR</div>', unsafe_allow_html=True)
else:
    st.markdown(confianza_html(confianza), unsafe_allow_html=True)

# Metricas clave
c1, c2, c3 = st.columns(3)
with c1:
    if sst_temp:
        st.metric("Temperatura", f"{float(sst_temp):.1f}C", help="SST L4 CMEMS")
    else:
        st.metric("Temperatura", "Sin dato")

with c2:
    surg_pct = int(ind_surg * 100)
    delta_surg = "Alta" if ind_surg >= 0.70 else "Moderada" if ind_surg >= 0.40 else "Baja"
    st.metric("Surgencia", f"{surg_pct}%", delta=delta_surg)

with c3:
    color_ola = "normal" if swh <= 1.0 else "inverse" if swh > 1.5 else "off"
    st.metric("Oleaje", f"{swh:.1f}m", delta="OK" if swh <= 1.5 else "ADVERSO",
              delta_color="normal" if swh <= 1.5 else "inverse")

# Sensores activos
st.markdown("**Sensores activos hoy:**")
sensores = []
if sst_ok:    sensores.append(" SST sin nubes")
else:         sensores.append(" SST no disponible")
if era5_ok:   sensores.append(" Viento ERA5")
else:         sensores.append(" Viento no disponible")
if s2_ok:     sensores.append(" Sentinel-2 clorofila")
else:         sensores.append(" S2 sin dato (nubosidad) -- modo Teatro Fisico")
if s1_dias <= 3: sensores.append(f" SAR radar ({s1_dias}d)")
elif s1_dias <= 6: sensores.append(f" SAR radar ({s1_dias}d -- algo antiguo)")
else:         sensores.append(" SAR sin dato reciente")

for s in sensores:
    st.markdown(f"- {s}")

if not s2_ok:
    st.info(" **Modo Teatro Fisico activo** -- el sistema predice por estructura "
            "oceanografica (temperatura, surgencia, geometria costera), no por clorofila.")

st.markdown(f"*Ultima actualizacion: {hora_str} UTC*")

# ================================================================
# ZONAS POR SECTOR
# ================================================================
if kill_switch:
    st.error(" El oleaje supera 1.5m. No se recomienda salir hoy.")
    st.stop()

st.markdown("---")
st.markdown("### Zonas de pesca por sector")
st.caption("Toca las coordenadas para abrir en Google Maps")

# Ordenar sectores
orden_sectores = ['COSTERO', 'SUR', 'NORTE', 'OESTE']
sectores_presentes = [s for s in orden_sectores if s in df_rep['sector'].values]

for sector in sectores_presentes:
    df_sec = df_rep[df_rep['sector'] == sector].sort_values('rank_sector').reset_index(drop=True)
    if len(df_sec) == 0:
        continue

    st.markdown(f'<div class="sector-header">{sector_nombre(sector)}</div>', unsafe_allow_html=True)

    for _, zona in df_sec.iterrows():
        rol       = str(zona.get('rol', ''))
        lat       = float(zona.get('lat', 0))
        lon       = float(zona.get('lon', 0))
        dist      = float(zona.get('dist_km', 0))
        score_abs = float(zona.get('score_abs', 0))
        s_local   = str(zona.get('semaforo_local', ''))
        s_local_upper = s_local.upper()
        lat16     = zona.get('lat_T16', lat)
        lon16     = zona.get('lon_T16', lon)
        desp      = float(zona.get('desp_km', 0))
        direccion = str(zona.get('direccion', ''))
        rank      = int(zona.get('rank_sector', 0))

        # Tipo de zona
        if "NUCLEO" in rol.upper():
            clase = "zona-nucleo"
        elif "VARIANTE" in rol.upper():
            clase = "zona-variante"
        else:
            clase = "zona-trampa"

        # Semaforo del punto
        if "VERDE" in s_local_upper:
            icono = "VERDE"
        elif "AMARILLO" in s_local_upper:
            icono = "VERDE"
        else:
            icono = "VERDE"

        # Certeza basada en score absoluto
        if score_abs >= 0.60:
            certeza = "Alta"
        elif score_abs >= 0.45:
            certeza = "Media"
        else:
            certeza = "Baja -- modo fisico"

        link_ahora = coordenadas_link(lat, lon)
        link_16h   = coordenadas_link(float(lat16), float(lon16))

        with st.expander(f"{icono} {rol_label(rol)} -- {dist:.1f} km", expanded=(rank==1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Posicion ahora:**")
                st.markdown(f"[{lat:.4f}, {lon:.4f}]({link_ahora})")
            with c2:
                st.markdown(f"**En 16h ({direccion}):**")
                st.markdown(f"[{float(lat16):.4f}, {float(lon16):.4f}]({link_16h})")

            st.markdown(f"Distancia: **{dist:.1f} km** del muelle")
            st.markdown(f"Certeza local: **{certeza}**")
            st.markdown(f"Desplazamiento estimado: **{desp:.1f} km** hacia {direccion}")

            if "NUCLEO" in rol.upper():
                st.success("Primer lance recomendado en este sector")
            elif "VARIANTE" in rol.upper():
                st.info("Si el punto principal falla, prueba aqui")
            else:
                st.warning("Trampa fisica -- util si el frente se dispersa")

# ================================================================
# NOTAS PARA CHRISTIAN
# ================================================================
st.markdown("---")
st.markdown("###  Como usar estas zonas")
st.markdown("""
1. **Llega al sector** que mas te convenga segun combustible y tiempo
2. **Empieza por el punto principal** () -- es el de mayor probabilidad fisica
3. **Si no hay senales** (aves, color del agua) en 20-30 min, muevete a la Alternativa A o B
4. **Los Refugios** son zonas de trampa fisica -- utiles cuando el mar esta movido
5. **Las coordenadas en 16h** te dicen hacia donde se mueve el agua
""")

st.markdown("---")
st.markdown(f'<div class="footer">PredictaMAR Costero v1.2  '
            f'Datos: {fecha_str}  '
            f'Proyecto UNI Startup 2025</div>', unsafe_allow_html=True)
