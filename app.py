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
    "randy":      "Luciano1",
    "maik":       "Luciano2",
    "samantha":   "Luciano3",
    "Chorrillos1":"Christian",
    "pescador1":  "MarPeru1",
    "pescador2":  "MarPeru2",
    "pescador3":  "MarPeru3",
}

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario_actual" not in st.session_state:
    st.session_state["usuario_actual"] = ""

if not st.session_state["autenticado"]:
    st.title("🎣 PredictaMAR")
    st.caption("Sistema de prediccion de zonas de pesca artesanal — Copernicus Marine Service")
    st.divider()
    # Botones de acceso rapido
    col_inv, col_web = st.columns(2)
   col_inv, col_web = st.columns(2)
    with col_inv:
        if st.button("Acceso Startup Chile", use_container_width=True, type="primary"):
            st.session_state["autenticado"]    = True
            st.session_state["usuario_actual"] = "invitado"
            st.rerun()
    with col_web:
        st.link_button("Ver pagina web", "https://predictamaocenaia.lovable.app/",
                       use_container_width=True)
    st.divider()

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
        ws_feat  = gfile.worksheet("FEATURES_7D")
        ws_rules = gfile.worksheet("SPECIES_RULES")
        features      = pd.DataFrame(ws_feat.get_all_records())
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
    hoy      = ephem.now()
    prev_new = ephem.previous_new_moon(hoy)
    next_new = ephem.next_new_moon(hoy)
    ciclo    = next_new - prev_new
    trans    = (hoy - prev_new) / ciclo
    if trans < 0.25:   fase = 0
    elif trans < 0.50: fase = 1
    elif trans < 0.75: fase = 2
    else:              fase = 3
    emoji, nombre = FASES_LUNARES[fase]
    return emoji, nombre

# ── Puertos ───────────────────────────────────────────────────────
PUERTOS = {
    "MATARANI":   (-17.00, -72.10),
    "ILO":        (-17.64, -71.34),
    "MORRO_SAMA": (-17.98, -70.86),
    "TACNA":      (-18.00, -70.50),
    "PUCUSANA":   (-12.48, -76.80),
    "CHORRILLOS": (-12.18, -77.02),
    "CALLAO":     (-12.06, -77.15),
    "PAITA":      (-5.09,  -81.11),
    "CHIMBOTE":   (-9.08,  -78.59),
    "PISCO":      (-13.70, -76.20),
    "HUACHO":     (-11.10, -77.61),
    "SALAVERRY":  (-8.22,  -78.98),
    "HUANCHACO":  (-8.08,  -79.12),
    "MOLLENDO":   (-16.90, -72.01),
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
    "BONITO":    "🐠", "JUREL":      "🐟", "CABALLA":  "🐟",
    "POTA":      "🦑", "MERLUZA":    "🐟", "LORNA":    "🐟",
    "CABINZA":   "🐟",
}

# ── Haversine ─────────────────────────────────────────────────────
def haversine_nm(lat1, lon1, lat2, lon2):
    R    = 3440.065
    la1  = np.radians(lat1)
    lo1  = np.radians(lon1)
    la2  = np.radians(np.array(lat2, float))
    lo2  = np.radians(np.array(lon2, float))
    a    = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ── Dirección cardinal ────────────────────────────────────────────
def direccion_cardinal(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    if abs(dlat) < 1e-6 and abs(dlon) < 1e-6:
        return "misma zona"
    if abs(dlat) > abs(dlon):
        base = "norte" if dlat > 0 else "sur"
    else:
        base = "este" if dlon > 0 else "oeste"
    if abs(dlat) > 0.01 and abs(dlon) > 0.01:
        if dlat > 0 and dlon > 0:   base = "noreste"
        elif dlat > 0 and dlon < 0: base = "noroeste"
        elif dlat < 0 and dlon > 0: base = "sureste"
        else:                        base = "suroeste"
    return base

# ── Advección táctica (versión app sin NetCDF) ────────────────────
def calcular_adveccion_app(lat_punto, lon_punto, df_zona, horas_latencia=24):
    """
    Estima desplazamiento advectivo usando curr_mean_7d del cerebro.
    Aproxima dirección usando gradiente de SST como proxy de flujo dominante.
    """
    if horas_latencia <= 24:
        factor = 1.0
        confianza = "ALTA"
    elif horas_latencia <= 48:
        factor = 0.6
        confianza = "MEDIA"
    else:
        return lat_punto, lon_punto, 0.0, "BAJA"

    # Punto más cercano en el df para obtener velocidad de corriente
    if df_zona.empty:
        return lat_punto, lon_punto, 0.0, confianza

    df_zona = df_zona.copy()
    df_zona["_d"] = haversine_nm(lat_punto, lon_punto,
                                  df_zona["lat"].values,
                                  df_zona["lon"].values)
    vecinos = df_zona.nsmallest(5, "_d")

    curr_speed = float(vecinos["curr_mean_7d"].mean())
    if np.isnan(curr_speed) or curr_speed < 0.05:
        return lat_punto, lon_punto, 0.0, confianza

    # Estimar dirección dominante usando gradiente de SST
    # SST decrece hacia la costa (surgencia) → flujo predominante hacia el noroeste en HCS
    sst_vals = vecinos["sst_mean_7d"].values
    lat_vals  = vecinos["lat"].values
    lon_vals  = vecinos["lon"].values

    # Gradiente SST como proxy de dirección de corriente
    if len(vecinos) >= 3:
        dlat_sst = np.polyfit(lat_vals, sst_vals, 1)[0]
        dlon_sst = np.polyfit(lon_vals, sst_vals, 1)[0]
        # Corriente fluye de cálido a frío (gradiente negativo)
        vo_proxy = -dlat_sst * 0.1
        uo_proxy = -dlon_sst * 0.1
    else:
        # Default HCS: corriente hacia el noroeste
        vo_proxy = 0.05
        uo_proxy = -0.08

    # Normalizar a velocidad real
    mag = np.sqrt(vo_proxy**2 + uo_proxy**2)
    if mag > 0:
        vo_n = (vo_proxy / mag) * curr_speed
        uo_n = (uo_proxy / mag) * curr_speed
    else:
        vo_n, uo_n = 0.0, 0.0

    seg = horas_latencia * 3600 * factor
    nueva_lat = lat_punto + (vo_n * seg) / 111000
    nueva_lon = lon_punto + (uo_n * seg) / (111000 * np.cos(np.deg2rad(lat_punto)))

    dist = float(haversine_nm(lat_punto, lon_punto,
                               np.array([nueva_lat]),
                               np.array([nueva_lon]))[0] * 1.852)

    return nueva_lat, nueva_lon, dist, confianza

# ── Refinamiento con cerebro (equivalente OLCI en app) ────────────
def refinar_con_cerebro(lat_hotspot, lon_hotspot, df_zona, radio_grados=0.15):
    """
    Versión app de refinar_con_olci.
    Busca el cluster de mayor CHL dentro de radio en el cerebro.
    """
    mask = (
        (df_zona["lat"] >= lat_hotspot - radio_grados) &
        (df_zona["lat"] <= lat_hotspot + radio_grados) &
        (df_zona["lon"] >= lon_hotspot - radio_grados) &
        (df_zona["lon"] <= lon_hotspot + radio_grados)
    )
    zona = df_zona[mask].copy()
    if len(zona) < 3:
        return lat_hotspot, lon_hotspot, False

    chl_p90 = zona["chl_mean_7d"].quantile(0.90)
    cluster = zona[zona["chl_mean_7d"] >= chl_p90]
    if cluster.empty:
        return lat_hotspot, lon_hotspot, False

    # Centroide ponderado por CHL
    lat_c = float(np.average(cluster["lat"], weights=cluster["chl_mean_7d"]))
    lon_c = float(np.average(cluster["lon"], weights=cluster["chl_mean_7d"]))
    return lat_c, lon_c, True

# ── Macro filter COMPLETO (igual que Colab) ───────────────────────
def macro_filter(df):
    chl       = df["chl_mean_7d"].median()
    grad_sst  = df["grad_sst_mean_7d"].median()
    estab     = 1 - df["chl_cv_7d"].fillna(1).median()
    curr      = df["curr_mean_7d"].median()
    persist   = df["persistencia_score"].fillna(0).median() if "persistencia_score" in df.columns else 0.5
    dias_consec = df["chl_dias_consecutivos"].fillna(0).median() if "chl_dias_consecutivos" in df.columns else 0
    sla       = df["sla_mean_7d"].fillna(0).median() if "sla_mean_7d" in df.columns else 0
    ekman     = df["ekman_7d"].fillna(0).median() if "ekman_7d" in df.columns else 0
    grad_sla  = df["grad_sla_mean_7d"].fillna(0).median() if "grad_sla_mean_7d" in df.columns else 0

    sc_chl    = min(chl / 1.0, 1.0)
    sc_grad   = min(grad_sst / 0.05, 1.0)
    sc_estab  = max(estab, 0)
    sc_curr   = 1.0 if 0.1 <= curr <= 0.6 else 0.3
    sc_persist = float(persist)

    # SLA dinámico
    sla_p10 = df["sla_mean_7d"].quantile(0.10) if "sla_mean_7d" in df.columns else -0.05
    sla_p25 = df["sla_mean_7d"].quantile(0.25) if "sla_mean_7d" in df.columns else -0.02
    sla_p50 = df["sla_mean_7d"].quantile(0.50) if "sla_mean_7d" in df.columns else 0.0
    sla_p75 = df["sla_mean_7d"].quantile(0.75) if "sla_mean_7d" in df.columns else 0.05

    if sla < sla_p10:        sc_sla = 1.0
    elif sla < sla_p25:      sc_sla = 0.80
    elif sla < sla_p75:      sc_sla = 0.55
    else:                    sc_sla = 0.20

    sla_trend = df["sla_mean_7d"].mean() - sla_p75 if "sla_mean_7d" in df.columns else 0
    if sla_trend < -0.02:    sc_sla = min(sc_sla + 0.15, 1.0)
    if sla < sla_p25 and chl > 1.0: sc_sla = min(sc_sla + 0.10, 1.0)

    # Gradiente SLA
    grad_sla_p75 = df["grad_sla_mean_7d"].fillna(0).quantile(0.75) if "grad_sla_mean_7d" in df.columns else 0
    sc_grad_sla  = min(grad_sla / max(grad_sla_p75, 1e-6), 1.0)
    if grad_sla > grad_sla_p75 and sla < sla_p50:
        sc_sla = min(sc_sla + 0.12, 1.0)

    # Ekman — índice de surgencia
    ekman_p75 = df["ekman_7d"].fillna(0).quantile(0.75) if "ekman_7d" in df.columns else 0
    if ekman > ekman_p75:    sc_ekman = 1.0
    elif ekman > 0:          sc_ekman = 0.7
    elif ekman > -0.5:       sc_ekman = 0.4
    else:                    sc_ekman = 0.1

    macro_score = (0.30 * sc_chl + 0.16 * sc_grad + 0.16 * sc_persist +
                   0.12 * sc_sla + 0.10 * sc_ekman + 0.08 * sc_grad_sla +
                   0.05 * sc_estab + 0.03 * sc_curr)

    razones = []
    if chl < 0.5:            razones.append("clorofila baja — poca comida en el agua")
    if grad_sst < 0.02:      razones.append("sin frentes termicos — mar plano")
    if "sla_mean_7d" in df.columns and sla > sla_p75:
        razones.append("agua oceanica caliente — biomasa en profundidad")
    if persist < 0.3:        razones.append("zona inestable — CHL no persistente")
    if estab < 0.2:          razones.append("condiciones inestables en los ultimos 7 dias")
    if curr > 0.8:           razones.append("corrientes muy fuertes — dificil concentracion")
    if ekman < -1.0:         razones.append("viento del norte — sin surgencia activa esta semana")

    zona_surgencia = (dias_consec >= 5 and
                      ("sla_mean_7d" in df.columns and sla < sla_p25) and
                      ekman > 0)

    if macro_score >= 0.65:  semaforo = "VERDE"
    elif macro_score >= 0.55: semaforo = "AMARILLO"
    else:                    semaforo = "ROJO"

    return semaforo, macro_score, razones, zona_surgencia

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

    chl_min  = float(rule.get("chl_min",  0))
    chl_max  = float(rule.get("chl_max",  99))
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

    cv2  = d["curr_mean_7d"].values
    c_lo = 0.10
    c_hi = curr_max * 0.60
    sc2  = np.ones(len(d))
    sc2[cv2 < c_lo] = np.clip(0.5 + 0.5 * (cv2[cv2 < c_lo] / c_lo), 0, 1)
    sc2[(cv2 >= c_lo) & (cv2 <= c_hi)] = 1.0
    m_med = (cv2 > c_hi) & (cv2 <= curr_max)
    sc2[m_med] = np.clip(1 - 0.5 * (cv2[m_med] - c_hi) / (curr_max - c_hi), 0.5, 1.0)
    sc2[cv2 > curr_max] = np.clip(1 - (cv2[cv2 > curr_max] - curr_max) / curr_max, 0, 0.5)
    d["sc_curr"] = sc2

    sal_mid   = (sal_min + sal_max) / 2
    sal_range = max((sal_max - sal_min) / 2, 0.01)
    d["sc_sal"]  = np.clip(1 - np.abs(d["sal_mean_7d"] - sal_mid) / sal_range, 0, 1)
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
            "Chl":  chl_min <= r["chl_mean_7d"] <= chl_max,
            "SST":  sst_min <= r["sst_mean_7d"] <= sst_max,
            "Corr": r["curr_mean_7d"] <= curr_max,
            "Sal":  sal_min <= r["sal_mean_7d"] <= sal_max,
        }
        fl.append(f)
        failed = [k for k, v in f.items() if not v]
        al.append("Alerta: " + ", ".join(failed) + " fuera de rango" if failed else "")
    d["flags"]    = fl
    d["ok_count"] = [sum(f.values()) for f in fl]
    d["alert"]    = al
    return d

def score_a_indice(score, ok):
    if not np.isfinite(score):
        score = 0.0
    base = 30 + score * 50
    adj  = {4: 15, 3: 5, 2: -10, 1: -20, 0: -25}.get(min(ok, 4), -25)
    ind  = base + adj
    if ok <= 2:
        ind = min(ind, 49)
    return int(np.clip(ind, 30, 95))

def indice_a_semaforo(score):
    if score >= 0.65:
        return "🟢", "VERDE", "#1B5E20"
    elif score >= 0.55:
        return "🟡", "AMARILLO", "#F57F17"
    else:
        return "🔴", "ROJO", "#B71C1C"

# ── Tarjeta ROJA ──────────────────────────────────────────────────
def generar_tarjeta_roja_bytes(species, modo, razones):
    sp = species.strip().upper()
    fig, ax = plt.subplots(figsize=(5, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")
    fig.patch.set_facecolor("#FFEBEE"); ax.set_facecolor("#FFEBEE")
    ax.add_patch(FancyBboxPatch((0, 13.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor="#B71C1C", edgecolor="none"))
    ax.text(5, 14.55, "PREDICTAMAR  🔴  " + sp,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", fontfamily="monospace")
    ax.text(5, 14.05, "Zona: " + modo,
            ha="center", va="center", fontsize=8, color="white")
    ax.text(5, 12.5, "⚠ ZONA NO RECOMENDADA HOY",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="#B71C1C")
    ax.text(5, 11.8, "El mar no esta activo en tu zona.",
            ha="center", va="center", fontsize=10, color="#555555")
    ax.text(5, 11.2, "Salir hoy tiene alto riesgo\nde no encontrar pesca.",
            ha="center", va="center", fontsize=9, color="#555555")
    ax.plot([0.5, 9.5], [10.7, 10.7], color="#B71C1C", linewidth=1, alpha=0.4)
    ax.text(5, 10.3, "Por que?", ha="center", fontsize=10,
            fontweight="bold", color="#B71C1C")
    y = 9.7
    for r in razones:
        ax.text(1.0, y, "• " + r, fontsize=9, va="center", color="#333333")
        y -= 0.7
    ax.plot([0.5, 9.5], [y, y], color="#B71C1C", linewidth=1, alpha=0.3)
    ax.text(5, y - 0.5, "Proxima revision: manana",
            ha="center", fontsize=9, color="#888888", style="italic")
    ax.text(5, 0.5,
            "PredictaMAR · Sistema de Corriente de Humboldt · Peru",
            ha="center", fontsize=6.5, color="#888888", style="italic")
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor="#FFEBEE")
    plt.close()
    buf.seek(0)
    return buf

# ── Generar imagen (equivalente Colab con advección) ──────────────
def generar_imagen_bytes(idx, row, species, modo, fase_emoji, fase_nombre, zona_surgencia=False):
    sp    = species.strip().upper()
    ch, cb = COLORES.get(sp, ("#0D2B55", "#E8F4FF"))
    em    = EMOJIS.get(sp, "🎣")
    hr    = HORARIOS.get(sp, "05:00-08:00 / 16:00-18:30")
    fl    = row.get("flags", {})
    al    = row.get("alert", "")

    fig, ax = plt.subplots(figsize=(5, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 19)
    ax.axis("off")
    fig.patch.set_facecolor(cb)
    ax.set_facecolor(cb)

    # Header
    ax.add_patch(FancyBboxPatch((0, 17.5), 10, 1.5,
                  boxstyle="round,pad=0.1", facecolor=ch, edgecolor="none"))
    ax.text(5, 18.55, "PREDICTAMAR  " + em + "  " + sp,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")
    ax.text(5, 18.05, "Zona: " + modo,
            ha="center", va="center", fontsize=8, color="white")

    # Número
    ax.add_patch(plt.Circle((5, 16.8), 0.65, color=ch, zorder=3))
    ax.text(5, 16.8, str(idx + 1),
            ha="center", va="center", fontsize=18,
            fontweight="bold", color="white", zorder=4)

    # Índice operativo
    indice = int(row["prob"])
    ic     = "#1B5E20" if indice >= 70 else "#E65100" if indice >= 50 else "#B71C1C"
    ax.text(5, 15.8, "Indice operativo: " + str(indice) + "%",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=ic)

    # Surgencia / alerta
    if zona_surgencia:
        ax.text(5, 15.35, "⭐ ZONA DE SURGENCIA ACTIVA",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="#1B5E20")
    elif al:
        ax.text(5, 15.3, "  " + al,
                ha="center", va="center", fontsize=7.5,
                color="#B71C1C", fontstyle="italic")

    ax.plot([0.5, 9.5], [15.0, 15.0], color=ch, linewidth=1.5, alpha=0.4)

    # Coordenadas
    ax.text(0.5, 14.6, "📍", fontsize=11, va="center")
    olci_ok  = row.get("cerebro_refinado", False)
    lat_show = row.get("lat_refinada", row["lat"]) if olci_ok else row["lat"]
    lon_show = row.get("lon_refinada", row["lon"]) if olci_ok else row["lon"]
    coord_color = "#1B5E20" if olci_ok else "#212121"
    ax.text(1.3, 14.6,
            str(abs(lat_show))[:7] + "S,  " + str(abs(lon_show))[:7] + "W",
            fontsize=10, va="center", color=coord_color, fontweight="bold")

    # Radio activo estimado
    if olci_ok:
        ax.text(1.3, 14.15,
                "Zona activa estimada: radio aprox. 2 km  ✓ CHL cluster",
                fontsize=7.5, va="center", color="#1B5E20", fontstyle="italic")
    else:
        ax.text(1.3, 14.15,
                "Zona activa estimada: radio aprox. 6 km",
                fontsize=7.5, va="center", color="#1565C0", fontstyle="italic")

    # ── Guía táctica de advección ──────────────────────────────────
    adv_dist = float(row.get("adv_dist_km", 0))
    if adv_dist > 0.5:
        adv_lat  = float(row.get("adv_lat", row["lat"]))
        adv_lon  = float(row.get("adv_lon", row["lon"]))
        dir_txt  = direccion_cardinal(row["lat"], row["lon"], adv_lat, adv_lon)
        confianza = row.get("adv_confianza", "MEDIA")

        # Calcular punto guía a 6km en esa dirección
        fguia   = 6.0 / 111.0
        dlat    = adv_lat - row["lat"]
        dlon    = adv_lon - row["lon"]
        mag     = max(abs(dlat) + abs(dlon), 1e-9)
        glat    = row["lat"] + (dlat / mag) * fguia
        glon    = row["lon"] + (dlon / mag) * fguia / np.cos(np.deg2rad(row["lat"]))

        ax.text(0.5, 13.65, "🌊", fontsize=9, va="center")
        ax.text(1.3, 13.65,
                "Si no encuentra actividad, avance hacia el " + dir_txt,
                fontsize=7.5, va="center", color="#0D47A1")
        ax.text(0.5, 13.25, "🧭", fontsize=9, va="center")
        ax.text(1.3, 13.25,
                "Punto guia opcional:  " + str(abs(glat))[:6] + "S,  " +
                str(abs(glon))[:6] + "W  (" + confianza + ")",
                fontsize=7.5, va="center", color="#555555")
        y_vars = 12.8
    else:
        y_vars = 13.7

    ax.plot([0.5, 9.5], [y_vars - 0.2, y_vars - 0.2],
            color=ch, linewidth=1, alpha=0.3)

    # Variables oceanográficas
    vars_ = [
        ("🌡️", "Temp. superficial",  str(round(row["sst_mean_7d"],  2))  + " C",     fl.get("SST",  True)),
        ("🧂",  "Salinidad",          str(round(row["sal_mean_7d"],  2))  + " UPS",   fl.get("Sal",  True)),
        ("📈",  "Grad. termico",      str(round(row["grad_sst_mean_7d"], 4)) + " C/km", True),
        ("🌿",  "Clorofila-a",        str(round(row["chl_mean_7d"],  4))  + " mg/m3", fl.get("Chl",  True)),
        ("🌀",  "Corriente",          str(round(row["curr_mean_7d"], 2))  + " m/s",   fl.get("Corr", True)),
        ("📏",  "Distancia",          str(round(row["dist_km"],      1))  + " km",    True),
        (fase_emoji, "Fase lunar",    fase_nombre,                                    True),
    ]

    y = y_vars - 0.5
    for ev, lb, vl, ok in vars_:
        sc2 = "#1B5E20" if ok else "#B71C1C"
        sym = "OK" if ok else "NO"
        ax.text(0.5, y, ev,  fontsize=9,  va="center")
        ax.text(1.3, y, lb,  fontsize=8,  va="center", color="#555555")
        ax.text(9.5, y, vl,  fontsize=9,  va="center",
                ha="right", fontweight="bold", color="#212121")
        ax.text(9.8, y, sym, fontsize=7,  va="center",
                color=sc2, fontweight="bold")
        ax.plot([0.5, 9.5], [y - 0.32, y - 0.32],
                color="#CCCCCC", linewidth=0.5, alpha=0.6)
        y -= 0.65

    # Horario
    ax.add_patch(FancyBboxPatch((0.3, y - 0.2), 9.4, 0.6,
                  boxstyle="round,pad=0.1", facecolor=ch,
                  alpha=0.15, edgecolor=ch, linewidth=1))
    ax.text(0.5, y + 0.1, "⏰", fontsize=9, va="center")
    ax.text(1.3, y + 0.1, "Mejor hora:  " + hr,
            fontsize=8.5, va="center", color=ch, fontweight="bold")
    y -= 0.75

    # Condiciones
    ct = "  ".join([("OK " if v else "NO ") + k for k, v in fl.items()])
    ax.text(5, y, ct, ha="center", va="center", fontsize=8, color="#444444")
    y -= 0.45

    ax.plot([0.5, 9.5], [y + 0.15, y + 0.15], color=ch, linewidth=1, alpha=0.3)
    ax.text(5, y - 0.1,
            "PredictaMAR v5.2 · Copernicus Marine Service · Peru",
            ha="center", va="center", fontsize=6.5,
            color="#888888", style="italic")
    ax.text(5, y - 0.45,
            "Indice operativo — no es probabilidad estadistica",
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
st.caption("Sistema de prediccion de zonas de pesca artesanal — Copernicus Marine Service · Peru")

features, species_rules = cargar_cerebro()

if features is None:
    st.error("No se pudo cargar el cerebro. Verifica la conexion con Drive.")
    st.stop()

st.success("Datos cargados: " + str(len(features)) + " puntos oceanograficos")
st.divider()

# ── Formulario ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    especie = st.selectbox("🐟 Especie objetivo", options=list(species_rules["species"]))
with col2:
    radio_km = st.slider("📏 Radio de busqueda (km)", 20, 200, 80, 10)

modo_busqueda = st.radio(
    "📍 Modo de busqueda",
    ["Por puerto", "Entre dos puertos", "Por coordenadas"],
    horizontal=True,
)

puerto = puerto_desde = puerto_hasta = lat_input = lon_input = None

if modo_busqueda == "Por puerto":
    puerto = st.selectbox("Puerto de salida", list(PUERTOS.keys()))
elif modo_busqueda == "Entre dos puertos":
    col3, col4 = st.columns(2)
    with col3: puerto_desde = st.selectbox("Puerto origen",  list(PUERTOS.keys()))
    with col4: puerto_hasta = st.selectbox("Puerto destino", list(PUERTOS.keys()))
else:
    col5, col6 = st.columns(2)
    with col5: lat_input = st.number_input("Latitud (negativa)",  -22.0, -3.0,  -12.0, 0.01)
    with col6: lon_input = st.number_input("Longitud (negativa)", -85.0, -68.0, -77.0, 0.01)

top_n  = st.selectbox("Numero de puntos recomendados", [1, 2, 3, 4, 5], index=2)
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

    # ── MacroFilter COMPLETO ──────────────────────────────────────
    semaforo, macro_score, razones, zona_surgencia = macro_filter(df)
    sem_emoji, sem_label, sem_color = indice_a_semaforo(macro_score)
    surgencia_txt = " ⭐ SURGENCIA ACTIVA" if zona_surgencia else ""

    st.divider()
    st.subheader("📊 Resultados — " + especie)
    st.caption("Fase lunar hoy: " + fase_emoji + " " + fase_nombre)

    # Semáforo de zona
    st.markdown(
        f"""
        <div style="background-color:{sem_color}22; border-left: 5px solid {sem_color};
        padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
        <span style="font-size:1.3em; font-weight:bold; color:{sem_color};">
        {sem_emoji} ZONA {sem_label}{surgencia_txt}
        </span>
        <span style="font-size:0.9em; color:#555; margin-left:12px;">
        MacroScore: {macro_score:.2f}
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── ZONA ROJA — no salir hoy ──────────────────────────────────
    if semaforo == "ROJO":
        st.error("⚠️ ZONA NO RECOMENDADA HOY — El mar no está activo en tu zona.")
        if razones:
            st.markdown("**¿Por qué?**")
            for r in razones:
                st.markdown("• " + r)
        buf_roja = generar_tarjeta_roja_bytes(especie, modo, razones)
        st.image(buf_roja, use_column_width=True)
        buf_roja.seek(0)
        st.download_button(
            label="⬇️ Descargar tarjeta zona roja",
            data=buf_roja,
            file_name="predictamar_zona_roja.png",
            mime="image/png",
            use_container_width=True,
        )
        st.stop()

    # ── Calcular score ────────────────────────────────────────────
    rule = species_rules[species_rules["species"] == especie].iloc[0]
    df   = calcular_score(df, rule)
    df   = df.dropna(subset=["score"])
    df   = df[df["score"] > 0].sort_values("score", ascending=False)

    if df.empty:
        st.warning("Sin puntos con score valido.")
        st.stop()

    res           = df.head(top_n).reset_index(drop=True)
    res["prob"]   = res.apply(lambda r: score_a_indice(r["score"], r["ok_count"]), axis=1)
    res["dist_km"] = (res["dist_nm"] * 1.852).round(1)
    res = res.sort_values(["prob", "dist_km"], ascending=[False, True]).reset_index(drop=True)

    # ── Advección y refinamiento ──────────────────────────────────
    horas_latencia = 24  # valor por defecto; Colab lo calcula dinámicamente
    for i, row in res.iterrows():
        adv_lat, adv_lon, adv_dist, confianza = calcular_adveccion_app(
            row["lat"], row["lon"], df, horas_latencia
        )
        res.loc[i, "adv_lat"]       = adv_lat
        res.loc[i, "adv_lon"]       = adv_lon
        res.loc[i, "adv_dist_km"]   = adv_dist
        res.loc[i, "adv_confianza"] = confianza

        lat_r, lon_r, refinado = refinar_con_cerebro(row["lat"], row["lon"], df)
        res.loc[i, "lat_refinada"]      = lat_r
        res.loc[i, "lon_refinada"]      = lon_r
        res.loc[i, "cerebro_refinado"]  = refinado

    st.caption("ℹ️ Indice operativo — no es probabilidad estadistica")

    # ── Mostrar resultados ────────────────────────────────────────
    for i, row in res.iterrows():
        adv_dist = float(row.get("adv_dist_km", 0))
        adv_tag  = ""
        if adv_dist > 0.5:
            adv_lat  = float(row.get("adv_lat", row["lat"]))
            adv_lon  = float(row.get("adv_lon", row["lon"]))
            dir_txt  = direccion_cardinal(row["lat"], row["lon"], adv_lat, adv_lon)
            adv_tag  = " → " + str(round(adv_dist, 1)) + "km (" + row.get("adv_confianza", "MEDIA") + ")"

        with st.expander(
            "Punto " + str(i+1) + " — Indice: " + str(row["prob"]) + "% | " +
            str(round(abs(row["lat"]), 4)) + "S, " +
            str(round(abs(row["lon"]), 4)) + "W | " +
            str(row["dist_km"]) + " km" + adv_tag,
            expanded=True
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Temperatura",  str(round(row["sst_mean_7d"], 2))  + " C")
            c2.metric("🌿 Clorofila-a",  str(round(row["chl_mean_7d"], 4))  + " mg/m3")
            c3.metric("🌀 Corriente",    str(round(row["curr_mean_7d"], 2)) + " m/s")
            c4, c5, c6 = st.columns(3)
            c4.metric("🧂 Salinidad",     str(round(row["sal_mean_7d"], 2))  + " UPS")
            c5.metric("📈 Grad. termico", str(round(row["grad_sst_mean_7d"], 4)) + " C/km")
            c6.metric(fase_emoji + " Fase lunar", fase_nombre)
            st.caption("⏰ Mejor hora: " + HORARIOS.get(especie, ""))

            # Guía táctica en la UI
            if adv_dist > 0.5:
                adv_lat2 = float(row.get("adv_lat", row["lat"]))
                adv_lon2 = float(row.get("adv_lon", row["lon"]))
                dir_txt2 = direccion_cardinal(row["lat"], row["lon"], adv_lat2, adv_lon2)
                fguia    = 6.0 / 111.0
                dlat     = adv_lat2 - row["lat"]
                dlon     = adv_lon2 - row["lon"]
                mag      = max(abs(dlat) + abs(dlon), 1e-9)
                glat     = row["lat"] + (dlat / mag) * fguia
                glon     = row["lon"] + (dlon / mag) * fguia / np.cos(np.deg2rad(row["lat"]))

                st.info(
                    "🌊 Si no encuentra actividad, avance hacia el **" + dir_txt2 + "**\n\n"
                    "🧭 Punto guía opcional: **" + str(abs(glat))[:6] + "S, " +
                    str(abs(glon))[:6] + "W** (" + row.get("adv_confianza", "MEDIA") + ")"
                )

            if row.get("cerebro_refinado"):
                lat_r = float(row.get("lat_refinada", row["lat"]))
                lon_r = float(row.get("lon_refinada", row["lon"]))
                st.success(
                    "✓ Coordenada refinada por cluster CHL: **" +
                    str(abs(lat_r))[:7] + "S, " + str(abs(lon_r))[:7] + "W** — radio aprox. 2 km"
                )

            if row.get("alert"):
                st.warning(row["alert"])

            # Imagen descargable
            buf = generar_imagen_bytes(i, row, especie, modo,
                                       fase_emoji, fase_nombre, zona_surgencia)
            st.image(buf, use_column_width=True)
            buf.seek(0)
            st.download_button(
                label="⬇️ Descargar tarjeta punto " + str(i+1),
                data=buf,
                file_name="predictamar_" + especie + "_punto" + str(i+1) + ".png",
                mime="image/png",
                use_container_width=True,
            )

    st.divider()
    st.caption(
        "PredictaMAR v5.2 · Sistema de Corriente de Humboldt · Peru · "
        "Datos: Copernicus Marine Service (CMEMS) · "
        "Indice operativo — no es probabilidad estadistica"
    )
