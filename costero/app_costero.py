import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials as SACredentials
import json
import pytz
from datetime import datetime, timezone

# ================================================================
# MAREAS -- Componentes armonicos Callao/Lima
# ================================================================
COMPONENTES_CALLAO = {
    'M2': (0.381, 175.2, 28.9841),
    'S2': (0.107, 195.8, 30.0000),
    'N2': (0.078, 158.4, 28.4397),
    'K1': (0.058, 210.5, 15.0411),
    'O1': (0.042, 195.3, 13.9430),
}
EPOCH_MAREAS = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
LIMA_TZ = pytz.timezone('America/Lima')

def altura_marea(dt_utc):
    t_horas = (dt_utc - EPOCH_MAREAS).total_seconds() / 3600.0
    h = 0.0
    for amp, fase_deg, vel_deg_h in COMPONENTES_CALLAO.values():
        h += amp * np.cos(np.radians(vel_deg_h * t_horas - fase_deg))
    return round(h, 3)

def fase_marea(dt_utc):
    dt_a = datetime.fromtimestamp(dt_utc.timestamp() - 1800, tz=timezone.utc)
    dt_d = datetime.fromtimestamp(dt_utc.timestamp() + 1800, tz=timezone.utc)
    dhdt = (altura_marea(dt_d) - altura_marea(dt_a)) / 1.0
    h = altura_marea(dt_utc)
    if abs(dhdt) < 0.005:
        return ("PLEAMAR" if h > 0 else "BAJAMAR"), h, 0.02
    elif dhdt > 0:
        return "LLENANTE", h, 0.05
    else:
        return "VACIANTE", h, 0.00

def mareas_hoy():
    hoy = datetime.now(LIMA_TZ).date()
    resultado = {}
    for hora in [4, 5, 6, 7, 17, 18]:
        dt_lima = LIMA_TZ.localize(datetime(hoy.year, hoy.month, hoy.day, hora, 0, 0))
        dt_utc = dt_lima.astimezone(timezone.utc)
        fase, altura, bonus = fase_marea(dt_utc)
        flecha = "" if fase in ("LLENANTE","PLEAMAR") else ""
        resultado[hora] = {"fase": fase, "altura": altura, "bonus": bonus, "flecha": flecha}
    return resultado

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
    st.error("ADVERSO -- Oleaje peligroso. No recomendado salir.")
elif confianza == "ALTA":
    st.success("Condiciones fisicas favorables -- datos completos con confirmacion biologica")
elif confianza == "MEDIA":
    st.info("Condiciones fisicas estimadas -- modo Teatro Fisico activo (sin dato optico)")
else:
    st.warning("Datos limitados -- usar criterio propio y experiencia empirica")

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

# ================================================================
# MAREAS -- ventanas operacionales de Christian
# ================================================================
mareas = mareas_hoy()
st.subheader("Marea hoy -- ventanas de pesca")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    v = mareas[4]
    st.metric("4AM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")
with col_m2:
    v = mareas[5]
    st.metric("5AM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")
with col_m3:
    v = mareas[6]
    st.metric("6AM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")
with col_m4:
    v = mareas[7]
    st.metric("7AM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")
col_m5, col_m6, _, _ = st.columns(4)
with col_m5:
    v = mareas[17]
    st.metric("5PM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")
with col_m6:
    v = mareas[18]
    st.metric("6PM", f"{v['flecha']} {v['fase'][:4]}", f"{v['altura']:+.2f}m")

n_llen = sum(1 for h in [4,5,6,7] if mareas[h]['fase'] in ("LLENANTE","PLEAMAR"))
if n_llen >= 3:
    st.success("Madrugada: marea llenante en la mayor parte de la ventana -- condicion favorable para pejerrey de orilla")
elif n_llen >= 2:
    st.info("Madrugada mixta -- parte de la ventana con marea llenante")
else:
    st.warning("Madrugada con vaciante predominante -- evaluar salida en la tarde")

st.divider()

# ================================================================
# HORA DE LANCE -- recalculo de adveccion en tiempo real
# ================================================================
st.subheader("Hora de lance")
st.caption("Ingresa la hora a la que planeas lanzar la red. El sistema calculara donde estara el agua en ese momento.")

col_hora1, col_hora2 = st.columns([2,3])
with col_hora1:
    hora_lance = st.selectbox(
        "Hora del lance (Lima)",
        options=list(range(4, 22)),
        index=13,
        format_func=lambda h: f"{h:02d}:00 {'AM' if h < 12 else 'PM'} Lima"
    )

import pytz as _pytz
_lima_tz_app = _pytz.timezone('America/Lima')
_ahora_lima_app = datetime.now(_lima_tz_app)
_hora_actual_decimal = _ahora_lima_app.hour + _ahora_lima_app.minute / 60.0
_horas_hasta_lance = max(0.5, hora_lance - _hora_actual_decimal)
_desplazamiento_km = round(_horas_hasta_lance * 0.19 * 3.6, 1)

with col_hora2:
    if _horas_hasta_lance <= 0.5:
        st.warning(f"Lance muy proximo. Usando minimo 30 min de proyeccion.")
    else:
        st.info(f"{_horas_hasta_lance:.1f}h hasta las {hora_lance:02d}:00 | Agua se habra desplazado ~{_desplazamiento_km} km")

st.divider()
st.subheader("Zonas de pesca -- 3 sectores")
st.caption("Toca 'Ver en mapa' para abrir el pin en Google Maps. Guarda captura antes de salir.")

LAT_CHORRILLOS = -12.157
LON_CHORRILLOS = -77.021
orden = ["COSTERO", "SUR", "OESTE"]
nombres = {"COSTERO": "Orilla (0-6 km)", "SUR": "Sur - Morro Solar",
           "OESTE": "Mar abierto"}
dirs_esp = ["Norte","NorEste","Este","SurEste","Sur","SurOeste","Oeste","NorOeste"]

for sector in orden:
    df_sec = df[df["sector"] == sector].copy() if "sector" in df.columns else pd.DataFrame()
    if len(df_sec) == 0:
        continue
    # Ordenar por rank -- extraer numero del rank_sector
    def extract_rank(val):
        try:
            # Si es numero directo
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val).strip()
            # Si viene como datetime string tipo "2026-06-05 00:00:01" -- extraer ultimo digito
            if "-" in s and ":" in s and len(s) > 10:
                # Es un datetime -- el rank esta codificado en los segundos
                segundos = int(s.split(":")[-1].split(".")[0].strip())
                if 1 <= segundos <= 5:
                    return segundos
                return 99
            # Si viene como "1.0", "2.0" etc
            return int(float(s))
        except:
            return 99
    df_sec["rank_num"] = df_sec["rank_sector"].apply(extract_rank)
    df_sec = df_sec.sort_values("rank_num").reset_index(drop=True)

    with st.expander(f"{nombres.get(sector, sector)} -- {len(df_sec)} puntos", expanded=(sector=="COSTERO")):

        # Curva de marea 24 horas -- Christian ve como se mueve el mar hoy
        hoy = datetime.now(LIMA_TZ).date()
        horas_24 = list(range(24))
        alturas = []
        fases_24 = []
        for h in horas_24:
            dt_lima = LIMA_TZ.localize(datetime(hoy.year, hoy.month, hoy.day, h, 0, 0))
            dt_utc = dt_lima.astimezone(timezone.utc)
            alt = altura_marea(dt_utc)
            fas, _, _ = fase_marea(dt_utc)
            alturas.append(alt)
            fases_24.append(fas)

        import pandas as pd
        df_marea = pd.DataFrame({
            "Hora": [f"{h:02d}h" for h in horas_24],
            "Marea (m)": alturas
        }).set_index("Hora")
        st.caption("Marea hoy -- sube y baja del mar (24 horas)")
        st.line_chart(df_marea, height=120, width='stretch')

        # Mejor momento del dia segun marea
        max_idx = alturas.index(max(alturas))
        min_idx = alturas.index(min(alturas))
        hora_actual = datetime.now(LIMA_TZ).hour
        fase_ahora, alt_ahora, _ = fase_marea(
            LIMA_TZ.localize(datetime(hoy.year, hoy.month, hoy.day, hora_actual, 0, 0)).astimezone(timezone.utc))
        flecha = "(sube)" if fase_ahora in ("LLENANTE","PLEAMAR") else "(baja)"
        st.caption(f"Ahora ({hora_actual:02d}h): {flecha} {fase_ahora} {alt_ahora:+.2f}m  |  Pleamar: {max_idx:02d}h  |  Bajamar: {min_idx:02d}h")

        st.divider()
        for _, zona in df_sec.iterrows():
            lat = to_float(zona.get("lat", 0))
            lon = to_float(zona.get("lon", 0))
            if abs(lat) < 1 or abs(lon) < 1:
                continue
            dist = to_float(zona.get("dist_km", 0))
            rol = str(zona.get("rol", ""))
            # Correccion: rank_sector puede venir como datetime, string o numero
            rank_raw = zona.get("rank_sector", 0)
            try:
                # Si es datetime tipo "2026-01-01 00:00:02", extraer segundos
                s = str(rank_raw)
                if ":" in s and "-" in s:
                    # Es un datetime -- extraer el segundo campo de tiempo
                    rank = int(s.split(":")[-1].split(".")[0].strip())
                else:
                    rank = int(float(s))
                if rank == 0:
                    rank = 1
            except:
                rank = 1
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

            # Score local como porcentaje
            score_loc = to_float(zona.get("score_local", 0))
            score_pct = int(score_loc * 100)

            # Coordenada de adveccion -- calculada dinamicamente para hora de lance ingresada
            # Usa corrientes uo/vo del pipeline y proyecta a _horas_hasta_lance
            uo = to_float(zona.get("uo_medio", -0.19))
            vo = to_float(zona.get("vo_medio", 0.11))
            seg = _horas_hasta_lance * 3600.0
            import math as _math
            dlat = (vo * seg) / 111000.0
            dlon = (uo * seg) / (111000.0 * _math.cos(_math.radians(lat)))
            lat16 = round(lat + dlat, 4)
            lon16 = round(lon + dlon, 4)
            desp  = round(_math.sqrt(((lat16-lat)*111.0)**2 + ((lon16-lon)*111.0*_math.cos(_math.radians(lat)))**2), 1)
            # Direccion del desplazamiento
            angulo = _math.degrees(_math.atan2(lon16-lon, lat16-lat))
            _dirs = ["N","NE","E","SE","S","SO","O","NO"]
            dir_adv = _dirs[int((angulo + 22.5) / 45) % 8]

            gmap = f"https://www.google.com/maps?q={lat},{lon}&z=14"

            # Linea principal del punto
            st.markdown(f"**{label}** ({score_pct}%) | {dist:.1f} km | ~{tiempo} min | Hacia el {dir_s} | [Ver en mapa]({gmap})")

            # Coordenada de adveccion -- solo si esta dentro del radio operacional de Christian (7km)
            if desp > 0.5 and desp <= 7.0 and abs(lat16) > 1 and abs(lon16) > 1:
                gmap16 = f"https://www.google.com/maps?q={lat16},{lon16}&z=14"
                st.markdown(f"  Si no hay actividad: el agua se movio {desp:.1f} km al {dir_adv} -- [Ver punto sugerido]({gmap16})")

st.divider()
st.caption("Antes de salir: guarda captura de pantalla del mapa. No tendras senal en el mar.")
st.caption("PredictaMAR v2.1 -- UNI Startup 2025")
