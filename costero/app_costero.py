import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials as SACredentials
import json

st.set_page_config(page_title="PredictaMAR", page_icon=":fish:", layout="centered")

USUARIOS = {
    "christian": "chorrillos2025",
    "randy": "predictamar2025",
    "maik": "predictamar2025",
    "samantha": "predictamar2025",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("PredictaMAR Costero")
    st.caption("Puerto Chorrillos - Lima")
    usuario = st.text_input("Usuario")
    clave = st.text_input("Clave", type="password")
    if st.button("Entrar"):
        if usuario.lower() in USUARIOS and USUARIOS[usuario.lower()] == clave:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Usuario o clave incorrectos")
    st.stop()

@st.cache_data(ttl=300)
def cargar_datos():
    try:
        sa_info = json.loads(st.secrets["GOOGLE_SA_JSON"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = SACredentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["SHEET_ID"])
        df = pd.DataFrame(sh.worksheet("costero_reporte").get_all_records())
        return df, None
    except Exception as e:
        return None, str(e)

def to_float(v, d=0.0):
    try:
        f = float(v)
        return f if not np.isnan(f) else d
    except:
        return d

def to_bool(v):
    if isinstance(v, bool): return v
    return str(v).upper() in ("TRUE", "1", "YES")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("PredictaMAR Costero")
with col2:
    if st.button("Salir"):
        st.session_state.logged_in = False
        st.rerun()

if st.button("Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

df, error = cargar_datos()

if error or df is None or len(df) == 0:
    st.error("Sin datos. Verifica conexion.")
    st.stop()

fila = df.iloc[0]
swh = to_float(fila.get("swh_medio", 0.8))
kill = to_bool(fila.get("kill_switch", False))
sst = to_float(fila.get("sst_temp_medio", 0))
surg = to_float(fila.get("indice_surgencia", 0.5))
s2 = to_bool(fila.get("s2_bloom_ok", False))
hora = str(fila.get("hora_utc", ""))
confianza = str(fila.get("confianza", "MEDIA"))
uo = to_float(fila.get("uo_medio", 0))
vo = to_float(fila.get("vo_medio", 0))
mar_adverso = (swh > 1.5) or kill

st.divider()

if mar_adverso:
    st.error("ADVERSO -- NO SALIR HOY. Oleaje peligroso.")
elif confianza == "ALTA":
    st.success("BUENAS CONDICIONES -- Listo para salir")
else:
    st.warning("CONDICIONES MODERADAS -- Usar criterio propio")

c1, c2, c3 = st.columns(3)
c1.metric("Temperatura", f"{sst:.0f}C")
c2.metric("Surgencia", f"{int(surg*100)}%")
c3.metric("Oleaje", f"{swh:.1f}m")

dirs = ["N","NE","E","SE","S","SO","O","NO"]
ang = float(np.degrees(np.arctan2(uo, vo)))
dir_agua = dirs[int((ang + 22.5) / 45) % 8]
modo = "Nublado -- modo fisico activo" if not s2 else "Satelite optico activo"
st.info(f"{modo} | Agua se mueve al {dir_agua} | Datos: {hora} UTC")

if mar_adverso:
    st.stop()

st.divider()
st.subheader("Zonas de pesca -- 4 sectores")
st.caption("Toca 'Ver en mapa' para abrir el pin en Google Maps. Guarda captura antes de salir.")

LAT_CHORRILLOS = -12.157
LON_CHORRILLOS = -77.021
orden = ["COSTERO", "SUR", "NORTE", "OESTE"]
nombres = {"COSTERO": "Orilla (0-3 km)", "SUR": "Sur - Morro Solar",
           "NORTE": "Norte - Miraflores", "OESTE": "Mar abierto"}
dirs_esp = ["Norte","NorEste","Este","SurEste","Sur","SurOeste","Oeste","NorOeste"]

for sector in orden:
    df_sec = df[df["sector"] == sector].copy() if "sector" in df.columns else pd.DataFrame()
    if len(df_sec) == 0:
        continue
    df_sec["rank_num"] = pd.to_numeric(df_sec.get("rank_sector", 0), errors="coerce").fillna(99)
    df_sec = df_sec.sort_values("rank_num").reset_index(drop=True)

    with st.expander(f"{nombres.get(sector, sector)} -- {len(df_sec)} puntos", expanded=(sector=="COSTERO")):
        for _, zona in df_sec.iterrows():
            lat = to_float(zona.get("lat", 0))
            lon = to_float(zona.get("lon", 0))
            if abs(lat) < 1 or abs(lon) < 1:
                continue
            dist = to_float(zona.get("dist_km", 0))
            rol = str(zona.get("rol", ""))
            rank = int(to_float(zona.get("rank_sector", 0)))
            tiempo = int(dist / 15 * 60)

            dlat = lat - LAT_CHORRILLOS
            dlon = lon - LON_CHORRILLOS
            ang_s = float(np.degrees(np.arctan2(dlon, dlat)))
            dir_s = dirs_esp[int((ang_s + 22.5) / 45) % 8]

            if "NUCLEO" in rol.upper():
                label = f"[{rank}] PUNTO PRINCIPAL"
            elif "VARIANTE" in rol.upper():
                label = f"[{rank}] Alternativa"
            else:
                label = f"[{rank}] Refugio"

            gmap = f"https://www.google.com/maps?q={lat},{lon}&z=14"
            st.markdown(f"**{label}** | {dist:.1f} km | ~{tiempo} min | Hacia el {dir_s} | [Ver en mapa]({gmap})")

st.divider()
st.caption("Antes de salir: guarda captura de pantalla del mapa. No tendras senal en el mar.")
st.caption("PredictaMAR v2.1 -- UNI Startup 2025")
