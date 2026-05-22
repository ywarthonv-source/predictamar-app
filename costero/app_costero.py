# ================================================================
# PredictaMAR Costero v1.2 -- APP STREAMLIT PARA CHRISTIAN
# Puerto Chorrillos - Lima - Sistema Corriente de Humboldt
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime
import json

st.set_page_config(
    page_title="PredictaMAR Costero",
    page_icon=":fish:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container{padding:0.75rem 1rem;max-width:500px;margin:auto}
h1{font-size:1.5rem!important;margin-bottom:0.1rem}
h3{font-size:1.1rem!important;margin-bottom:0.25rem}
.pill-verde{background:#1D9E75;color:white;padding:10px 16px;border-radius:8px;
    font-weight:bold;font-size:1.1rem;text-align:center;margin:6px 0;display:block}
.pill-amarillo{background:#BA7517;color:white;padding:10px 16px;border-radius:8px;
    font-weight:bold;font-size:1.1rem;text-align:center;margin:6px 0;display:block}
.pill-rojo{background:#A32D2D;color:white;padding:10px 16px;border-radius:8px;
    font-weight:bold;font-size:1.1rem;text-align:center;margin:6px 0;display:block}
.pill-adverso{background:#2C2C2A;color:white;padding:10px 16px;border-radius:8px;
    font-weight:bold;font-size:1.1rem;text-align:center;margin:6px 0;display:block}
.sector-hdr{background:#0C447C;color:white;padding:8px 14px;border-radius:8px;
    font-weight:bold;font-size:1rem;margin:12px 0 6px}
.zona-nucleo{border-left:4px solid #1D9E75;background:#F4FAF8;
    border-radius:0 8px 8px 0;padding:10px 12px;margin:4px 0}
.zona-variante{border-left:4px solid #185FA5;background:#F0F6FB;
    border-radius:0 8px 8px 0;padding:10px 12px;margin:4px 0}
.zona-trampa{border-left:4px solid #888780;background:#F5F4F1;
    border-radius:0 8px 8px 0;padding:10px 12px;margin:4px 0}
.dato-row{display:flex;justify-content:space-between;font-size:0.85rem;
    padding:2px 0;border-bottom:0.5px solid #E8E7E0}
.dato-row:last-child{border-bottom:none}
.dato-lbl{color:#5F5E5A}
.dato-val{font-weight:500}
.alert-info{background:#E6F1FB;border-left:3px solid #185FA5;padding:8px 12px;
    border-radius:0 6px 6px 0;font-size:0.85rem;margin:6px 0}
.footer{font-size:0.7rem;color:#888780;text-align:center;margin-top:1.5rem}
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOGIN SIMPLE
# ================================================================
USUARIOS = {
    "christian": "chorrillos2025",
    "randy":     "predictamar2025",
    "maik":      "predictamar2025",
    "samantha":  "predictamar2025",
}

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.markdown("# PredictaMAR Costero")
        st.markdown("**Puerto Chorrillos - Lima**")
        st.markdown("---")
        with st.form("login_form"):
            st.markdown("### Ingresa tus datos")
            usuario = st.text_input("Usuario", placeholder="tu usuario")
            clave   = st.text_input("Clave", type="password", placeholder="tu clave")
            btn     = st.form_submit_button("Entrar")
            if btn:
                if usuario.lower() in USUARIOS and USUARIOS[usuario.lower()] == clave:
                    st.session_state.logged_in = True
                    st.session_state.usuario   = usuario.lower()
                    st.rerun()
                else:
                    st.error("Usuario o clave incorrectos")
        return False
    return True

if not check_login():
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
        df_ipo  = pd.DataFrame(sh.worksheet("costero_ipo").get_all_records())
        return df_rep, df_ipo, None
    except Exception as e:
        return None, None, str(e)

# ================================================================
# HELPERS
# ================================================================
def to_bool(val):
    if isinstance(val, bool): return val
    if str(val).upper() in ("TRUE","1","YES"): return True
    return False

def to_float(val, default=0.0):
    try: return float(val)
    except: return default

def semaforo_pill(s):
    s = str(s).upper()
    if "ADVERSO" in s:
        return '<span class="pill-adverso">ADVERSO -- NO SALIR HOY</span>'
    if "VERDE" in s:
        return '<span class="pill-verde">VERDE -- CONDICION FAVORABLE</span>'
    if "AMARILLO" in s:
        return '<span class="pill-amarillo">AMARILLO -- CONDICION POSIBLE</span>'
    return '<span class="pill-rojo">ROJO -- CONDICION BAJA</span>'

def rol_icon(rol):
    rol = str(rol).upper()
    if "NUCLEO"    in rol: return "[1] PUNTO PRINCIPAL"
    if "VARIANTE_1" in rol: return "[2] Alternativa A"
    if "VARIANTE_2" in rol: return "[3] Alternativa B"
    if "TRAMPA_1"  in rol: return "[4] Refugio 1"
    if "TRAMPA_2"  in rol: return "[5] Refugio 2"
    return rol

def rol_clase(rol):
    rol = str(rol).upper()
    if "NUCLEO"   in rol: return "zona-nucleo"
    if "VARIANTE" in rol: return "zona-variante"
    return "zona-trampa"

def sector_nombre(s):
    return {
        "COSTERO": "Orilla (0-3 km)",
        "SUR":     "Sur -- Morro Solar",
        "NORTE":   "Norte -- Miraflores",
        "OESTE":   "Oeste -- Mar abierto"
    }.get(str(s).upper(), s)

def maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def certeza_texto(score_abs):
    s = to_float(score_abs)
    if s >= 0.60: return "Alta"
    if s >= 0.45: return "Media"
    return "Baja (modo fisico)"

# ================================================================
# ENCABEZADO
# ================================================================
col_t, col_b = st.columns([3,1])
with col_t:
    st.markdown("# PredictaMAR Costero")
    st.markdown("**Puerto Chorrillos -- Lima**")
with col_b:
    if st.button("Actualizar"):
        st.cache_data.clear()
        st.rerun()

# ================================================================
# CARGAR DATOS
# ================================================================
df_rep, df_ipo, error = cargar_datos()

if error:
    st.error(f"Error al cargar datos: {error}")
    st.stop()

if df_rep is None or len(df_rep) == 0:
    st.warning("Sin datos. El pipeline corre a las 3AM y 3PM Lima.")
    st.stop()

# Datos globales del dia
fila         = df_rep.iloc[0]
fecha_str    = str(fila.get("fecha",""))
hora_str     = str(fila.get("hora_utc",""))
confianza    = str(fila.get("confianza","MEDIA"))
kill_switch  = to_bool(fila.get("kill_switch", False))
swh          = to_float(fila.get("swh_medio", 0.8))
sst_temp     = fila.get("sst_temp_medio", None)
ind_surg     = to_float(fila.get("indice_surgencia", 0.5))
s2_ok        = to_bool(fila.get("s2_bloom_ok", False))
sst_ok       = to_bool(fila.get("sst_ok", False))
era5_ok      = to_bool(fila.get("era5_ok", False))
s1_dias_raw  = fila.get("s1_dias", 99)
s1_dias      = int(to_float(s1_dias_raw, 99))
uo           = to_float(fila.get("uo_medio", 0))
vo           = to_float(fila.get("vo_medio", 0))
bonus_trampa = to_float(fila.get("bonus_trampa", 0))

# Determinar semaforo general -- CORRECCION: kill_switch es string "False"/"True"
mar_adverso = (swh > 1.5) or kill_switch

# ================================================================
# ESTADO DEL MAR
# ================================================================
st.markdown("---")
st.markdown("### Estado del mar hoy")

if mar_adverso:
    st.markdown('<span class="pill-adverso">ADVERSO -- NO SALIR HOY</span>', unsafe_allow_html=True)
    st.error(f"El oleaje es {swh:.1f}m. Supera el limite operacional de 1.5m.")
else:
    if confianza == "ALTA":
        st.markdown('<span class="pill-verde">MAR OPERACIONAL -- Buenas condiciones</span>', unsafe_allow_html=True)
    elif confianza == "MEDIA":
        st.markdown('<span class="pill-amarillo">MAR OPERACIONAL -- Confianza media</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-rojo">MAR OPERACIONAL -- Datos limitados</span>', unsafe_allow_html=True)

# Metricas
c1, c2, c3 = st.columns(3)
with c1:
    temp_val = f"{to_float(sst_temp):.1f}C" if sst_temp else "Sin dato"
    st.metric("Temperatura", temp_val,
              delta="Rango pejerrey: 17-22C" if sst_temp else None)
with c2:
    st.metric("Surgencia", f"{int(ind_surg*100)}%",
              delta="Alta" if ind_surg >= 0.70 else "Moderada",
              delta_color="normal" if ind_surg >= 0.50 else "inverse")
with c3:
    st.metric("Oleaje", f"{swh:.1f}m",
              delta="OK" if swh <= 1.0 else "Moderado" if swh <= 1.5 else "ADVERSO",
              delta_color="normal" if swh <= 1.0 else "off" if swh <= 1.5 else "inverse")

# Corrientes
vel_corriente = np.sqrt(uo**2 + vo**2) * 100  # cm/s
dirs = ["N","NE","E","SE","S","SO","O","NO"]
angulo_cor = float(np.degrees(np.arctan2(uo, vo)))
dir_cor = dirs[int((angulo_cor + 22.5) / 45) % 8]

st.markdown(f"""
<div class="alert-info">
<b>Corrientes hoy:</b> {vel_corriente:.1f} cm/s hacia el {dir_cor} --
En 16 horas el agua se desplaza ~{vel_corriente*0.576:.1f} km en esa direccion
</div>
""", unsafe_allow_html=True)

# Sensores
st.markdown("**Sensores activos:**")
sensores_html = ""
sensores_html += f"<li>{'[OK]' if sst_ok else '[NO]'} Temperatura del mar (SST L4 CMEMS)</li>"
sensores_html += f"<li>{'[OK]' if era5_ok else '[NO]'} Viento y surgencia (ERA5)</li>"
if s2_ok:
    sensores_html += "<li>[OK] Sentinel-2 clorofila -- datos biologicos directos</li>"
else:
    sensores_html += "<li>[--] Sentinel-2 sin dato -- nubosidad -- modo Teatro Fisico activo</li>"
if s1_dias <= 3:
    sensores_html += f"<li>[OK] Radar SAR Sentinel-1 -- {s1_dias} dias de antiguedad</li>"
elif s1_dias <= 6:
    sensores_html += f"<li>[~~] Radar SAR -- {s1_dias} dias de antiguedad (algo antiguo)</li>"
else:
    sensores_html += "<li>[NO] Radar SAR sin dato reciente</li>"

st.markdown(f"<ul>{sensores_html}</ul>", unsafe_allow_html=True)

if not s2_ok:
    st.markdown("""
<div class="alert-info">
<b>Modo Teatro Fisico:</b> Sin imagen de satelite optico por nubes.
El sistema predice por temperatura del mar, viento y geometria costera.
Los puntos son los mejores disponibles con la informacion de hoy.
</div>
""", unsafe_allow_html=True)

st.markdown(f"*Datos de: {fecha_str} -- Actualizado: {hora_str} UTC*")

# ================================================================
# ZONAS POR SECTOR
# ================================================================
if mar_adverso:
    st.error("No se muestran zonas porque el mar esta adverso hoy.")
    st.stop()

st.markdown("---")
st.markdown("### Zonas de pesca -- 4 sectores")
st.caption("Toca las coordenadas para abrir en Google Maps")

orden_sectores = ["COSTERO", "SUR", "NORTE", "OESTE"]
sectores_presentes = [s for s in orden_sectores if s in df_rep["sector"].values]

if len(sectores_presentes) == 0:
    st.warning("Sin datos de sectores. Espera la proxima actualizacion.")
    st.stop()

for sector in sectores_presentes:
    df_sec = df_rep[df_rep["sector"] == sector].copy()
    # Convertir rank_sector a numero para ordenar
    df_sec["rank_num"] = pd.to_numeric(df_sec["rank_sector"], errors="coerce").fillna(99)
    df_sec = df_sec.sort_values("rank_num").reset_index(drop=True)

    if len(df_sec) == 0:
        continue

    st.markdown(f'<div class="sector-hdr">{sector_nombre(sector)} -- {len(df_sec)} puntos</div>',
                unsafe_allow_html=True)

    for _, zona in df_sec.iterrows():
        rol       = str(zona.get("rol",""))
        lat       = to_float(zona.get("lat",0))
        lon       = to_float(zona.get("lon",0))
        dist      = to_float(zona.get("dist_km",0))
        score_abs = to_float(zona.get("score_abs",0))
        score_loc = to_float(zona.get("score_local",0))
        sem_local = str(zona.get("semaforo_local",""))
        lat16     = to_float(zona.get("lat_T16", lat))
        lon16     = to_float(zona.get("lon_T16", lon))
        desp      = to_float(zona.get("desp_km",0))
        direccion = str(zona.get("direccion",""))
        rank      = int(to_float(zona.get("rank_sector",0)))

        clase   = rol_clase(rol)
        rotulo  = rol_icon(rol)
        certeza = certeza_texto(score_abs)

        # Semaforo local del punto
        sem_upper = sem_local.upper()
        if "VERDE" in sem_upper:   icono_sem = "[V]"
        elif "AMARILLO" in sem_upper: icono_sem = "[~]"
        else:                         icono_sem = "[x]"

        link_ahora = maps_link(lat, lon)
        link_16h   = maps_link(lat16, lon16)

        # Score local como porcentaje
        score_pct = int(score_loc * 100)

        contenido = f"""
<div class="{clase}">
<div style="font-weight:bold;font-size:0.95rem;margin-bottom:6px">
  {icono_sem} {rotulo}
</div>
<div class="dato-row">
  <span class="dato-lbl">Posicion ahora</span>
  <span class="dato-val"><a href="{link_ahora}" target="_blank">{lat:.4f}, {lon:.4f}</a></span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Posicion en 16h</span>
  <span class="dato-val"><a href="{link_16h}" target="_blank">{lat16:.4f}, {lon16:.4f}</a></span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Distancia del muelle</span>
  <span class="dato-val">{dist:.1f} km</span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Temperatura del mar</span>
  <span class="dato-val">{f"{to_float(sst_temp):.1f}C" if sst_temp else "Sin dato"}</span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Surgencia activa</span>
  <span class="dato-val">{int(ind_surg*100)}% -- {"Alta" if ind_surg >= 0.70 else "Moderada" if ind_surg >= 0.40 else "Baja"}</span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Probabilidad local</span>
  <span class="dato-val">{score_pct}% en este sector</span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Certeza</span>
  <span class="dato-val">{certeza}</span>
</div>
<div class="dato-row">
  <span class="dato-lbl">Desplazamiento agua</span>
  <span class="dato-val">{desp:.1f} km hacia {direccion} en 16h</span>
</div>
<div style="font-size:0.8rem;color:#5F5E5A;margin-top:6px;padding-top:4px;border-top:0.5px solid #E8E7E0">
"""
        if "NUCLEO" in rol.upper():
            contenido += "Primer lance recomendado. Mayor probabilidad fisica del sector."
        elif "VARIANTE" in rol.upper():
            contenido += "Si el punto principal no da resultado, prueba aqui."
        else:
            contenido += "Trampa fisica -- util si el cardumen se disperso del frente."

        contenido += "</div></div>"
        st.markdown(contenido, unsafe_allow_html=True)

# ================================================================
# COMO USAR
# ================================================================
st.markdown("---")
st.markdown("### Como usar estas zonas")
st.markdown("""
1. **Elige el sector** segun combustible y tiempo disponible
2. **Ve al Punto 1** (principal) -- es donde el modelo calcula mayor probabilidad
3. **Si no hay senales** en 20-30 min (aves, color del agua), muevete al punto siguiente
4. **Los Refugios** son trampas fisicas costeras -- utiles cuando el mar esta movido
5. **Las coordenadas en 16h** indican hacia donde se mueve el agua
""")

# ================================================================
# LOGOUT
# ================================================================
st.markdown("---")
col_l, col_r = st.columns([3,1])
with col_r:
    if st.button("Salir"):
        st.session_state.logged_in = False
        st.rerun()

st.markdown(f'<div class="footer">PredictaMAR Costero v1.2 -- {fecha_str} -- UNI Startup 2025</div>',
            unsafe_allow_html=True)
