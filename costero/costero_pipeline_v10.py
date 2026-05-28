# ================================================================
# PredictaMAR Costero v1.2 -- PIPELINE SECTORIZACION TACTICA
# Puerto Chorrillos - 0-20 km - Sistema Corriente de Humboldt
#
# Cambios v1.2:
#   - 4 sectores geograficos: Costero, Norte, Sur, Oeste
#   - Z-score local por sector (normalizacion relativa)
#   - Top 5 por sector con roles tacticos:
#       Punto 1: nucleo del frente (max score local)
#       Puntos 2-3: variantes del frente (min 500m de separacion)
#       Puntos 4-5: trampas estaticas moduladas por condiciones
#   - Total: hasta 20 puntos distribuidos en 4 sectores
#   - Score correcto: contribuciones aditivas acotadas [0,1]
#   - Adveccion corregida: coordenadas reales T8h y T16h
# ================================================================

import sys, os, json
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime, timedelta, timezone
import copernicusmarine
import ee
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# MODULO DE MAREAS -- Componentes armonicos Callao/Lima
# Fuente: IHO Tidal Constituent Database / IMARPE / literatura HCS
# Marea semidiurna: dos pleamares y dos bajamares por dia
# Pejerrey entra con marea llenante hacia los roquedales de orilla
# Bertrand et al. 2008, Gutierrez et al. 2012
# ================================================================
import pytz

COMPONENTES_CALLAO = {
    'M2': (0.381, 175.2, 28.9841),  # Principal lunar semidiurna
    'S2': (0.107, 195.8, 30.0000),  # Principal solar semidiurna
    'N2': (0.078, 158.4, 28.4397),  # Lunar eliptica mayor
    'K1': (0.058, 210.5, 15.0411),  # Lunisolar diurna
    'O1': (0.042, 195.3, 13.9430),  # Principal lunar diurna
}
EPOCH_MAREAS = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
LIMA_TZ = pytz.timezone('America/Lima')

def altura_marea_callao(dt_utc):
    t_horas = (dt_utc - EPOCH_MAREAS).total_seconds() / 3600.0
    h = 0.0
    for nombre, (amp, fase_deg, vel_deg_h) in COMPONENTES_CALLAO.items():
        fase_rad = np.radians(vel_deg_h * t_horas - fase_deg)
        h += amp * np.cos(fase_rad)
    return round(h, 3)

def fase_marea_callao(dt_utc, delta_min=30):
    """
    Calcula fase de marea usando derivada numerica dh/dt
    Retorna: (fase_str, altura_m, dhdt_m_h, bonus_score)
    fase_str: LLENANTE | VACIANTE | PLEAMAR | BAJAMAR
    bonus: +0.05 LLENANTE, +0.02 PLEAMAR, 0.00 resto
    """
    dt_antes   = datetime.fromtimestamp(dt_utc.timestamp() - delta_min*60, tz=timezone.utc)
    dt_despues = datetime.fromtimestamp(dt_utc.timestamp() + delta_min*60, tz=timezone.utc)
    h_antes    = altura_marea_callao(dt_antes)
    h_ahora    = altura_marea_callao(dt_utc)
    h_despues  = altura_marea_callao(dt_despues)
    dhdt = (h_despues - h_antes) / (2 * delta_min / 60)

    if abs(dhdt) < 0.005:
        fase  = "PLEAMAR" if h_ahora > 0 else "BAJAMAR"
        bonus = 0.02
    elif dhdt > 0:
        fase  = "LLENANTE"
        bonus = 0.05
    else:
        fase  = "VACIANTE"
        bonus = 0.00

    return fase, h_ahora, round(dhdt, 4), bonus

def calcular_ventanas_christian(fecha_hoy):
    """
    Calcula fase de marea en las ventanas operacionales de Christian:
    Madrugada: 4AM, 5AM, 6AM, 7AM Lima
    Tarde:     5PM, 6PM Lima
    Retorna dict con fase, altura, bonus y flecha visual por hora
    """
    ventanas = {}
    for hora in [4, 5, 6, 7, 17, 18]:
        dt_lima = LIMA_TZ.localize(
            datetime(fecha_hoy.year, fecha_hoy.month, fecha_hoy.day, hora, 0, 0))
        dt_utc  = dt_lima.astimezone(timezone.utc)
        fase, altura, dhdt, bonus = fase_marea_callao(dt_utc)
        flecha = "" if fase in ("LLENANTE","PLEAMAR") else ""
        ventanas[hora] = {
            'fase':   fase,
            'altura': altura,
            'dhdt':   dhdt,
            'bonus':  bonus,
            'flecha': flecha,
            'etiqueta': f"{flecha} {fase} ({altura:+.2f}m)"
        }
    return ventanas


print("=" * 60)
print("PredictaMAR Costero v1.2 -- SECTORIZACION TACTICA")
print(f"Inicio: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from costero_config import *

# -- Credenciales
CMEMS_USER = os.environ["CMEMS_USER"]
CMEMS_PASS = os.environ["CMEMS_PASS"]
CDS_KEY    = os.environ["CDS_KEY"]
SHEET_ID   = os.environ["SHEET_ID"]
GOOGLE_SA  = os.environ["GOOGLE_SA_JSON"]

with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
    f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {CDS_KEY}\n")

sa_info  = json.loads(GOOGLE_SA)
scopes   = ["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]
sa_creds = SACredentials.from_service_account_info(sa_info, scopes=scopes)
gc       = gspread.authorize(sa_creds)
sh       = gc.open_by_key(SHEET_ID)
print(f"Sheets OK: {sh.title}")

copernicusmarine.login(username=CMEMS_USER, password=CMEMS_PASS,
                       force_overwrite=True)
print("CMEMS OK")

gee_creds = ee.ServiceAccountCredentials(sa_info["client_email"],
                                          key_data=sa_info["private_key"])
ee.Initialize(gee_creds)
print("GEE OK")
sys.stdout.flush()

# -- Fechas
FECHA_HOY     = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
FECHA_HOY_STR = FECHA_HOY.strftime("%Y-%m-%d")
FECHA_BLOOM   = FECHA_HOY - timedelta(days=LAG_CHL_DIAS)
FECHA_SURG    = FECHA_HOY - timedelta(days=T_SURGENCIA_DIAS)
DRIVE_BASE    = "/tmp/predictamar_costero"
os.makedirs(f"{DRIVE_BASE}/raw", exist_ok=True)

print(f"Hoy: {FECHA_HOY_STR} | Bloom: {FECHA_BLOOM.strftime('%Y-%m-%d')} | Surgencia: {FECHA_SURG.strftime('%Y-%m-%d')}")

# Calcular mareas para ventanas de Christian
VENTANAS_MAREA = calcular_ventanas_christian(FECHA_HOY)
print("Mareas Chorrillos hoy:")
for hora, v in VENTANAS_MAREA.items():
    ventana_nombre = "madrugada" if hora < 12 else "tarde"
    print(f"  {hora:02d}:00 Lima [{ventana_nombre}]: {v['etiqueta']} | bonus={v['bonus']:.2f}")

# Bonus global de marea: promedio ponderado de las ventanas
# Ventanas madrugada tienen mayor peso (Christian sale mas temprano)
bonus_marea_madrugada = np.mean([VENTANAS_MAREA[h]['bonus'] for h in [4,5,6,7]])
bonus_marea_tarde     = np.mean([VENTANAS_MAREA[h]['bonus'] for h in [17,18]])
bonus_marea_global    = round(bonus_marea_madrugada * 0.7 + bonus_marea_tarde * 0.3, 3)
print(f"  Bonus marea global: madrugada={bonus_marea_madrugada:.3f} tarde={bonus_marea_tarde:.3f} global={bonus_marea_global:.3f}")
sys.stdout.flush()

AOI = ee.Geometry.Rectangle([LON_MIN_C, LAT_MIN_C, LON_MAX_C, LAT_MAX_C])

def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.sqrt(dlat**2 + dlon**2))

# ================================================================
# SECTORIZACION GEOGRAFICA DE CHORRILLOS
# Basada en morfologia costera y experiencia operacional
#
# Sector COSTERO: 0-3 km -- rompiente, pejerrey de orilla
# Sector NORTE:   3-15 km al norte -- frente a Miraflores
# Sector SUR:     3-20 km al sur -- sotavento Morro Solar
# Sector OESTE:   12-20 km -- aguas abiertas HCS
#
# La sectorizacion usa distancia + angulo desde el muelle
# ================================================================
def asignar_sector(lat, lon, dist):
    """
    Asigna sector geografico basado en posicion respecto al muelle.
    Muelle Chorrillos: -12.157, -77.021
    Norte de Lima (Miraflores): latitudes menores (mas negativas en valor)
    Sur (Morro Solar, Lurin): latitudes mayores
    """
    # Angulo desde el muelle
    dlat = lat - LAT_CHORRILLOS
    dlon = lon - LON_CHORRILLOS

    # Sector costero ampliado a 6km -- captura zona operacional de Christian
    if dist <= 6.0:
        return "COSTERO"

    # Norte: hacia Miraflores (lat mas al norte = valor menos negativo)
    if dlat > 0.02:  # mas de ~2 km al norte del muelle
        return "NORTE"

    # Sur: hacia Morro Solar y Lurin (lat mas al sur = valor mas negativo)
    if dlat < -0.02:  # mas de ~2 km al sur del muelle
        return "SUR"

    # Oeste: aguas abiertas (lejos de la costa)
    if dist > 12.0:
        return "OESTE"

    # Zona intermedia -- asignar por predominancia
    if abs(dlat) > abs(dlon):
        return "SUR" if dlat < 0 else "NORTE"
    return "OESTE"

# ================================================================
# FUENTE 1 -- SST L4 CMEMS (surgencia T-7d)
# ================================================================
print("\n[1/8] SST L4 CMEMS -- surgencia T-7d...")
SST_DISPONIBLE = False
sst_grad_medio = 0.5
sst_temp_medio = None

try:
    sst_path = f"{DRIVE_BASE}/raw/sst_l4.nc"
    if os.path.exists(sst_path): os.remove(sst_path)
    copernicusmarine.subset(
        dataset_id           = CMEMS_SST_L4,
        variables            = ["analysed_sst"],
        minimum_latitude     = LAT_MIN_C,
        maximum_latitude     = LAT_MAX_C,
        minimum_longitude    = LON_MIN_C,
        maximum_longitude    = LON_MAX_C,
        start_datetime       = (FECHA_SURG - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime         = FECHA_SURG.strftime("%Y-%m-%dT23:59:59"),
        output_filename      = "sst_l4.nc",
        output_directory     = f"{DRIVE_BASE}/raw",
        username             = CMEMS_USER,
        password             = CMEMS_PASS,
        disable_progress_bar = True
    )
    ds_sst   = xr.open_dataset(sst_path)
    sst_raw  = ds_sst['analysed_sst'].values
    sst_C    = sst_raw - 273.15
    sst_mean = np.nanmean(sst_C, axis=0) if sst_C.ndim == 3 else sst_C
    gy, gx   = np.gradient(np.where(np.isnan(sst_mean), 0, sst_mean))
    sst_grad = np.sqrt(gx**2 + gy**2)
    p5, p95  = np.nanpercentile(sst_grad, 5), np.nanpercentile(sst_grad, 95)
    sst_grad_medio = float(np.nanmean(np.clip((sst_grad - p5) / (p95 - p5 + 1e-9), 0, 1)))
    sst_temp_medio = float(np.nanmean(sst_mean))
    SST_DISPONIBLE = True
    print(f"  SST OK -- T: {sst_temp_medio:.1f}C | grad: {sst_grad_medio:.3f}")
except Exception as e:
    print(f"  SST error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 2 -- ERA5 VIENTO (proxy surgencia)
# ================================================================
print("\n[2/8] ERA5 viento -- proxy surgencia...")
ERA5_DISPONIBLE  = False
indice_surgencia = 0.5

try:
    era5_path = f"{DRIVE_BASE}/raw/era5_viento.nc"
    if os.path.exists(era5_path): os.remove(era5_path)
    meses = list(set([FECHA_SURG.strftime("%m"), FECHA_HOY.strftime("%m")]))
    dias  = list(set([(FECHA_SURG + timedelta(days=i)).strftime("%d")
                      for i in range(T_SURGENCIA_DIAS + 1)]))
    c_api = cdsapi.Client()
    c_api.retrieve(
        ERA5_DATASET,
        {"product_type": "reanalysis",
         "variable": ERA5_VARIABLES,
         "year":  FECHA_HOY.strftime("%Y"),
         "month": meses, "day": dias,
         "time":  ["00:00", "06:00", "12:00", "18:00"],
         "area":  [LAT_MAX_C+0.5, LON_MIN_C-0.5, LAT_MIN_C-0.5, LON_MAX_C+0.5],
         "format": "netcdf"},
        era5_path
    )
    ds_era5 = xr.open_dataset(era5_path)
    u = float(np.nanmean(ds_era5['u10'].values))
    v = float(np.nanmean(ds_era5['v10'].values))
    indice_surgencia = float(np.clip(max(0, v) / (abs(v) + abs(u) + 1e-9), 0, 1))
    ERA5_DISPONIBLE = True
    print(f"  ERA5 OK -- U: {u:.2f} V: {v:.2f} | ind_surg: {indice_surgencia:.3f}")
except Exception as e:
    print(f"  ERA5 error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 3 -- OLEAJE GEE (kill-switch + penalizacion mezcla)
# ================================================================
print("\n[3/8] Oleaje -- condicion del mar...")
OLEAJE_DISPONIBLE   = False
swh_medio           = 0.8
kill_switch_activo  = False
penalizacion_mezcla = 0.0

try:
    fecha_ola_ini = (FECHA_HOY - timedelta(days=1)).strftime("%Y-%m-%d")
    waves_col = (ee.ImageCollection(GEE_WAVES)
                   .filterBounds(AOI)
                   .filterDate(fecha_ola_ini, FECHA_HOY_STR)
                   .select(['VHM0']))
    if waves_col.size().getInfo() > 0:
        waves_img = waves_col.sort('system:time_start', False).first()
        swh_stats = waves_img.reduceRegion(
            reducer  = ee.Reducer.mean(),
            geometry = AOI,
            scale    = 9000
        ).getInfo()
        swh_val = swh_stats.get('VHM0')
        if swh_val is not None:
            swh_medio = float(swh_val)
            OLEAJE_DISPONIBLE = True
            if swh_medio > SWH_KILL_SWITCH:
                kill_switch_activo = True
                print(f"  OLEAJE ADVERSO -- SWH: {swh_medio:.2f}m -- KILL SWITCH")
            else:
                penalizacion_mezcla = -float(np.clip((swh_medio / SWH_MEZCLA_MAX) * 0.10, 0, 0.10))
                print(f"  Oleaje OK -- SWH: {swh_medio:.2f}m | pen: {penalizacion_mezcla:.3f}")
    else:
        print("  Oleaje sin imagen reciente")
except Exception as e:
    print(f"  Oleaje error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 4 -- CORRIENTES CMEMS (adveccion)
# ================================================================
print("\n[4/8] Corrientes CMEMS -- adveccion...")
CORRIENTES_DISPONIBLE = False
uo_medio = 0.0
vo_medio = 0.0

try:
    cur_path = f"{DRIVE_BASE}/raw/corrientes.nc"
    if os.path.exists(cur_path): os.remove(cur_path)
    copernicusmarine.subset(
        dataset_id           = CMEMS_CORRIENTES_NRT,
        variables            = ["uo", "vo"],
        minimum_latitude     = LAT_MIN_C,
        maximum_latitude     = LAT_MAX_C,
        minimum_longitude    = LON_MIN_C,
        maximum_longitude    = LON_MAX_C,
        start_datetime       = (FECHA_HOY - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime         = FECHA_HOY.strftime("%Y-%m-%dT23:59:59"),
        minimum_depth        = 0.49,
        maximum_depth        = 0.51,
        output_filename      = "corrientes.nc",
        output_directory     = f"{DRIVE_BASE}/raw",
        username             = CMEMS_USER,
        password             = CMEMS_PASS,
        disable_progress_bar = True
    )
    ds_cur = xr.open_dataset(cur_path)
    uo_medio = float(np.nanmean(ds_cur['uo'].values))
    vo_medio = float(np.nanmean(ds_cur['vo'].values))
    CORRIENTES_DISPONIBLE = True
    print(f"  Corrientes OK -- uo: {uo_medio:.4f} vo: {vo_medio:.4f} m/s")
except Exception as e:
    print(f"  Corrientes error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 5 -- SENTINEL-2 (bloom T-4d, solo cielo despejado)
# ================================================================
print("\n[5/8] Sentinel-2 -- bloom T-4d...")
S2_DISPONIBLE    = False
Fc_s2            = 0.0
chl_bloom_imagen = None

try:
    def mask_s2_nubes(image):
        scl  = image.select('SCL')
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)

    fecha_b_ini = (FECHA_BLOOM - timedelta(days=3)).strftime("%Y-%m-%d")
    fecha_b_fin = (FECHA_BLOOM + timedelta(days=2)).strftime("%Y-%m-%d")
    s2_col = (ee.ImageCollection(GEE_S2)
                .filterBounds(AOI)
                .filterDate(fecha_b_ini, fecha_b_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .select(['B3', 'B4', 'SCL'])
                .map(mask_s2_nubes))
    if s2_col.size().getInfo() > 0:
        s2_img    = s2_col.sort('system:time_start', False).first()
        s2_ts     = datetime.fromtimestamp(s2_img.get('system:time_start').getInfo() / 1000)
        dias_diff = abs((FECHA_BLOOM - s2_ts).days)
        b3 = s2_img.select('B3').toFloat().divide(10000)
        b4 = s2_img.select('B4').toFloat().divide(10000)
        chl_bloom_imagen = b3.divide(b4.add(1e-6)).rename('score')
        S2_DISPONIBLE = True
        Fc_s2 = max(0.0, 1.0 - dias_diff / 5.0)
        print(f"  S2 OK -- fecha: {s2_ts.strftime('%Y-%m-%d')} | Fc: {Fc_s2:.2f}")
    else:
        print("  S2 sin dato -- nubosidad Lima")
except Exception as e:
    print(f"  S2 error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 6 -- SENTINEL-1 SAR
# ================================================================
print("\n[6/8] Sentinel-1 SAR...")
S1_DISPONIBLE = False
s1_norm       = None
Fc_s1         = 0.0
s1_ant        = 99

try:
    fecha_s1_ini = (FECHA_HOY - timedelta(days=FC_S1_MAX_DIAS + 2)).strftime("%Y-%m-%d")
    s1_col = (ee.ImageCollection(GEE_S1)
                .filterBounds(AOI)
                .filterDate(fecha_s1_ini, FECHA_HOY_STR)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select(['VV']))
    if s1_col.size().getInfo() > 0:
        s1_img    = s1_col.sort('system:time_start', False).first()
        s1_ts     = datetime.fromtimestamp(s1_img.get('system:time_start').getInfo() / 1000)
        s1_ant    = (FECHA_HOY - s1_ts).days
        s1_smooth = s1_img.focal_mean(radius=2, kernelType='circle', units='pixels')
        s1_grad   = s1_smooth.gradient().abs().reduce(ee.Reducer.max())
        s1_norm   = s1_grad.unitScale(
            s1_grad.reduceRegion(ee.Reducer.percentile([5]),  AOI, 100).values().get(0),
            s1_grad.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('score')
        S1_DISPONIBLE = True
        Fc_s1 = max(0.0, 1.0 - s1_ant / FC_S1_MAX_DIAS)
        # Filtro de viento para SAR: si viento fuera de rango 3-8 m/s
        # la senal SAR es poco confiable (calma = falsos slicks, viento alto = ruido)
        if ERA5_DISPONIBLE:
            vel_viento = float(np.sqrt(u**2 + v**2)) if 'u' in dir() else 0.0
            if vel_viento < 3.0 or vel_viento > 8.0:
                Fc_s1_original = Fc_s1
                Fc_s1 = Fc_s1 * 0.50  # degradar 50% fuera del rango optimo
                print(f"  S1 OK -- antiguedad: {s1_ant}d | Fc: {Fc_s1:.2f} (degradado 50% por viento={vel_viento:.1f}m/s fuera de 3-8m/s)")
            else:
                print(f"  S1 OK -- antiguedad: {s1_ant}d | Fc: {Fc_s1:.2f} (viento={vel_viento:.1f}m/s en rango optimo SAR)")
        else:
            print(f"  S1 OK -- antiguedad: {s1_ant}d | Fc: {Fc_s1:.2f}")
    else:
        print("  S1 sin dato reciente")
except Exception as e:
    print(f"  S1 error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 7+8 -- ALOS-2 + BATIMETRIA + GEOMETRIA COSTERA
# ================================================================
print("\n[7/8] ALOS-2 + batimetria + geometria costera...")
ALOS2_DISPONIBLE  = False
alos2_norm        = None
Fc_alos2          = 0.0
gebco_slope       = None
geometria_costera = None

try:
    fecha_alos_ini = (FECHA_HOY - timedelta(days=FC_ALOS2_MAX_DIAS + 5)).strftime("%Y-%m-%d")
    alos2_col = (ee.ImageCollection(GEE_ALOS2)
                   .filterBounds(AOI)
                   .filterDate(fecha_alos_ini, FECHA_HOY_STR)
                   .select(['HH']))
    if alos2_col.size().getInfo() > 0:
        alos2_img = alos2_col.sort('system:time_start', False).first()
        alos2_ts  = datetime.fromtimestamp(alos2_img.get('system:time_start').getInfo() / 1000)
        alos2_ant = (FECHA_HOY - alos2_ts).days
        alos2_s   = alos2_img.focal_mean(radius=3, kernelType='circle', units='pixels')
        alos2_g   = alos2_s.gradient().abs().reduce(ee.Reducer.max())
        alos2_norm = alos2_g.unitScale(
            alos2_g.reduceRegion(ee.Reducer.percentile([5]),  AOI, 100).values().get(0),
            alos2_g.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('score')
        ALOS2_DISPONIBLE = True
        Fc_alos2 = max(0.0, 1.0 - alos2_ant / FC_ALOS2_MAX_DIAS)
        print(f"  ALOS-2 OK -- antiguedad: {alos2_ant}d | Fc: {Fc_alos2:.2f}")
    else:
        print("  ALOS-2 sin dato reciente")
except Exception as e:
    print(f"  ALOS-2 error: {e}")

try:
    gebco        = ee.Image(GEE_GEBCO).select(GEE_GEBCO_BAND).clip(AOI)
    gebco_slope  = ee.Terrain.slope(gebco).rename('score')
    gebco_smooth = gebco.focal_mean(radius=3, kernelType='circle', units='pixels')
    grad_x       = gebco_smooth.gradient().select('x')
    grad_y       = gebco_smooth.gradient().select('y')
    grad_mag     = grad_x.pow(2).add(grad_y.pow(2)).sqrt()
    grad2_x      = grad_mag.gradient().select('x')
    grad2_y      = grad_mag.gradient().select('y')
    curv_bati    = grad2_x.pow(2).add(grad2_y.pow(2)).sqrt()
    geometria_costera = curv_bati.unitScale(
        curv_bati.reduceRegion(ee.Reducer.percentile([5]),  AOI, 500).values().get(0),
        curv_bati.reduceRegion(ee.Reducer.percentile([95]), AOI, 500).values().get(0)
    ).rename('score')
    print("  Batimetria + geometria OK")
except Exception as e:
    print(f"  Batimetria error: {e}")
sys.stdout.flush()

# ================================================================
# PESOS ADAPTATIVOS HORIZONTALES
# ================================================================
print("\nCalculando pesos adaptativos...")

W_activos = {}
if SST_DISPONIBLE:    W_activos["sst_grad"]    = W_BASE["sst_grad"]
if ERA5_DISPONIBLE:   W_activos["era5_viento"] = W_BASE["era5_viento"]
if S2_DISPONIBLE:     W_activos["chl_bloom"]   = W_BASE["chl_bloom"] * Fc_s2
if S1_DISPONIBLE:     W_activos["s1_sar"]      = W_BASE["s1_sar"] * Fc_s1
if ALOS2_DISPONIBLE:  W_activos["alos2"]       = W_BASE["alos2"] * Fc_alos2
if gebco_slope:       W_activos["batimetria"]  = W_BASE["batimetria"]
if geometria_costera: W_activos["geometria"]   = W_BASE["geometria"]

total_w = sum(W_activos.values())
if total_w > 0:
    W_activos = {k: v / total_w for k, v in W_activos.items()}

n_bio = sum([SST_DISPONIBLE, ERA5_DISPONIBLE, S2_DISPONIBLE])
n_fis = sum([S1_DISPONIBLE, ALOS2_DISPONIBLE, bool(gebco_slope), bool(geometria_costera)])
# Confianza corregida: ALTA solo con datos biologicos directos (S2) activos
# En modo Teatro Fisico (sin S2) la confianza es MEDIA por definicion
# porque los pesos no estan calibrados empiricamente
if S2_DISPONIBLE and n_bio >= 2 and n_fis >= 3:
    CONFIANZA = "ALTA"
elif n_bio >= 2 and n_fis >= 2:
    CONFIANZA = "MEDIA"
elif n_bio >= 1 and n_fis >= 2:
    CONFIANZA = "MEDIA"
else:
    CONFIANZA = "BAJA"

print(f"  Bio: {n_bio} | Fisico: {n_fis} | Confianza: {CONFIANZA}")
print(f"  Pesos: {', '.join([f'{k[:6]}:{v:.2f}' for k, v in W_activos.items()])}")
sys.stdout.flush()

# -- Contribucion SST: aditiva acotada (gradiente termico local)
contrib_sst = float(np.clip(sst_grad_medio * 0.10, 0, 0.10)) if SST_DISPONIBLE else 0.0

# -- ERA5 como MULTIPLICADOR REGIONAL (no contribucion aditiva)
# M_viento entre 0.80 (calma, sin surgencia) y 1.15 (surgencia alta activa)
# Potencia las estructuras reales que SAR y batimetria detectan
# pero si no hay estructura, el viento solo NO inventa un punto
if ERA5_DISPONIBLE:
    M_viento = float(np.clip(0.80 + indice_surgencia * 0.35, 0.80, 1.15))
else:
    M_viento = 1.0  # neutro si no hay dato de viento
print(f"  M_viento (multiplicador ERA5): {M_viento:.3f} (ind_surg={indice_surgencia:.3f})")

# -- Modificador de trampas estaticas segun condiciones dinamicas
# Si hay surgencia activa, las zonas de retencion geometrica se potencian
bonus_trampa = 0.0
if SST_DISPONIBLE and ERA5_DISPONIBLE:
    if indice_surgencia >= 0.70 and sst_grad_medio >= 0.40:
        bonus_trampa = 0.08   # Surgencia alta + gradiente termico = trampas activas
    elif indice_surgencia >= 0.50:
        bonus_trampa = 0.04   # Surgencia moderada
print(f"  Contrib SST: +{contrib_sst:.3f} | M_viento: x{M_viento:.3f} | Bonus trampa: +{bonus_trampa:.3f}")
sys.stdout.flush()

# ================================================================
# CALCULO DEL MICROSCORE CON GRILLA VARIABLE
# Zona costera 0-3 km: paso 0.0009 grados (~100m)
# Zona media 3-10 km:  paso 0.0027 grados (~300m)
# Zona abierta 10-20km: paso 0.0045 grados (~500m)
# ================================================================
# ================================================================
# MASCARA DE TIERRA -- linea de costa Costa Verde Lima
# ================================================================
COSTA_VERDE_MASCARA = [
    (-12.100, -12.080, -77.055),
    (-12.120, -12.100, -77.050),
    (-12.140, -12.120, -77.045),
    (-12.160, -12.140, -77.040),
    (-12.180, -12.160, -77.038),
    (-12.200, -12.180, -77.035),
]

def punto_en_tierra(lat, lon):
    for lat_min, lat_max, lon_max in COSTA_VERDE_MASCARA:
        if lat_min <= lat <= lat_max:
            if lon >= lon_max:
                return True
    return False

print("\nGenerando grilla variable...")

puntos_flat = []
# Grilla densa costera
lats_c = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0009)
lons_c = np.arange(LON_MIN_C, LON_MAX_C, 0.0009)
for la in lats_c:
    for lo in lons_c:
        if punto_en_tierra(float(la), float(lo)):
            continue  # excluir puntos en tierra
        d = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, float(la), float(lo))
        if d <= 6.0 and d >= RADIO_ORILLA_KM:
            puntos_flat.append((float(la), float(lo), d))

# Grilla media
lats_m = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0027)
lons_m = np.arange(LON_MIN_C, LON_MAX_C, 0.0027)
for la in lats_m:
    for lo in lons_m:
        if punto_en_tierra(float(la), float(lo)):
            continue  # excluir puntos en tierra
        d = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, float(la), float(lo))
        if d > 6.0 and d <= 10.0:
            puntos_flat.append((float(la), float(lo), d))

# Grilla abierta
lats_a = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0045)
lons_a = np.arange(LON_MIN_C, LON_MAX_C, 0.0045)
for la in lats_a:
    for lo in lons_a:
        if punto_en_tierra(float(la), float(lo)):
            continue  # excluir puntos en tierra
        d = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, float(la), float(lo))
        if d > 10.0 and d <= RADIO_MAX_KM:
            puntos_flat.append((float(la), float(lo), d))

print(f"  Puntos en grilla: {len(puntos_flat)} (costero + medio + abierto)")

# Organizar puntos por sector para muestreo representativo
from collections import defaultdict
puntos_por_sector = defaultdict(list)
for lat, lon, dist in puntos_flat:
    s = asignar_sector(lat, lon, dist)
    puntos_por_sector[s].append((lat, lon, dist))

# Muestra representativa: max 150 puntos por sector
MAX_POR_SECTOR = 150
puntos_muestra = []
for sector_key, pts in puntos_por_sector.items():
    if len(pts) > MAX_POR_SECTOR:
        indices = np.linspace(0, len(pts)-1, MAX_POR_SECTOR, dtype=int)
        pts_sel = [pts[i] for i in indices]
    else:
        pts_sel = pts
    puntos_muestra.extend(pts_sel)
    print(f"  Sector {sector_key}: {len(pts)} puntos -> {len(pts_sel)} muestreados")

puntos_flat = puntos_muestra
print(f"  Total a evaluar: {len(puntos_flat)} puntos")
sys.stdout.flush()

def adveccion_punto(lat, lon, horas):
    if not CORRIENTES_DISPONIBLE:
        return round(float(lat), 4), round(float(lon), 4)
    segundos = float(horas) * 3600.0
    dlat = (float(vo_medio) * segundos) / 111000.0
    dlon = (float(uo_medio) * segundos) / (111000.0 * float(np.cos(np.radians(float(lat)))))
    return round(float(lat) + dlat, 4), round(float(lon) + dlon, 4)

# ================================================================
# CALCULO DEL SCORE POR BATCH
# ================================================================
print("\nCalculando MicroScore por sector...")
resultados = []

for i in range(0, len(puntos_flat), 100):
    batch = puntos_flat[i:i+100]
    features = []
    for lat, lon, dist in batch:
        features.append(ee.Feature(
            ee.Geometry.Point([lon, lat]),
            {'lat': lat, 'lon': lon, 'dist_km': dist}
        ))
    if not features:
        continue

    try:
        fc    = ee.FeatureCollection(features)
        capas = []

        if S1_DISPONIBLE and s1_norm:
            capas.append(s1_norm.toFloat().clamp(0, 1).multiply(float(W_activos.get("s1_sar", 0))))
        if ALOS2_DISPONIBLE and alos2_norm:
            capas.append(alos2_norm.toFloat().clamp(0, 1).multiply(float(W_activos.get("alos2", 0))))
        if gebco_slope:
            capas.append(gebco_slope.toFloat().unitScale(0, 30).clamp(0, 1).multiply(float(W_activos.get("batimetria", 0))))
        if geometria_costera:
            capas.append(geometria_costera.toFloat().clamp(0, 1).multiply(float(W_activos.get("geometria", 0))))
        if S2_DISPONIBLE and chl_bloom_imagen:
            capas.append(chl_bloom_imagen.toFloat().unitScale(0.8, 1.5).clamp(0, 1).multiply(float(W_activos.get("chl_bloom", 0))))
        if not capas:
            continue

        capas_norm = [c.toFloat().rename('score') for c in capas]
        score_img  = ee.ImageCollection(capas_norm).sum().rename('score')
        res = score_img.reduceRegions(
            collection = fc,
            reducer    = ee.Reducer.mean(),
            scale      = 30
        ).getInfo()

        for feat in res['features']:
            p = feat['properties']
            if p.get('mean') is not None:
                lat_p      = float(p['lat'])
                lon_p      = float(p['lon'])
                score_base = float(p['mean'])
                sector     = asignar_sector(lat_p, lon_p, float(p['dist_km']))

                # Score absoluto con nueva arquitectura:
                # score_base se multiplica por M_viento (modulador regional ERA5)
                # luego se suman contribuciones locales, marea y bonus empirico
                score_con_viento = float(np.clip(score_base * M_viento, 0.0, 1.0))

                # -- Bonus empirico capa Christian (condicional a surgencia)
                bonus_empirico = 0.0
                for zona_emp in CHRISTIAN_ZONAS_EMPIRICAS:
                    dist_emp = float(np.sqrt(
                        ((lat_p - zona_emp["lat"]) * 111.0)**2 +
                        ((lon_p - zona_emp["lon"]) * 111.0 * np.cos(np.radians(lat_p)))**2
                    ))
                    # Activar bonus si punto esta a menos de 1km de zona empirica
                    # Y la surgencia supera el umbral minimo
                    if dist_emp < 1.0 and indice_surgencia >= zona_emp["surgencia_min"]:
                        bonus_empirico = max(bonus_empirico, zona_emp["bonus"])

                score_abs = float(np.clip(
                    score_con_viento + contrib_sst + penalizacion_mezcla + bonus_marea_global + bonus_empirico,
                    0.0, 1.0
                ))

                # Adveccion corregida
                lat_8h,  lon_8h  = adveccion_punto(lat_p, lon_p, ADV_HORAS_T8)
                lat_16h, lon_16h = adveccion_punto(lat_p, lon_p, ADV_HORAS_T16)
                dist_16h = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, lat_16h, lon_16h)
                desp_km  = dist_km(lat_p, lon_p, lat_16h, lon_16h)
                angulo   = float(np.degrees(np.arctan2(lon_16h - lon_p, lat_16h - lat_p)))
                dirs     = ["N","NE","E","SE","S","SO","O","NO"]
                dir_txt  = dirs[int((angulo + 22.5) / 45) % 8]

                resultados.append({
                    'lat':          lat_p,
                    'lon':          lon_p,
                    'dist_km':      round(float(p['dist_km']), 1),
                    'sector':       sector,
                    'score_abs':    round(score_abs, 4),
                    'score_base':   round(score_base, 4),
                    'lat_T8':       lat_8h,
                    'lon_T8':       lon_8h,
                    'lat_T16':      lat_16h,
                    'lon_T16':      lon_16h,
                    'dist_T16':     round(dist_16h, 1),
                    'desp_km':      round(desp_km, 2),
                    'direccion':    dir_txt,
                })
    except Exception as e:
        print(f"  Batch {i} error: {e}")
        continue

print(f"  Puntos con score: {len(resultados)}")
sys.stdout.flush()

# ================================================================
# Z-SCORE LOCAL POR SECTOR + SELECCION TOP 5 TACTICO
# ================================================================
print("\nAplicando Z-score local por sector...")

if not resultados:
    print("Sin resultados -- pipeline termina sin exportar")
    sys.exit(0)

df = pd.DataFrame(resultados)

# Z-score local: normaliza dentro de cada sector
for sector in df['sector'].unique():
    mask = df['sector'] == sector
    mu   = df.loc[mask, 'score_abs'].mean()
    sig  = df.loc[mask, 'score_abs'].std() + 1e-6
    df.loc[mask, 'z_score_local'] = (df.loc[mask, 'score_abs'] - mu) / sig

# Normalizar z_score_local a [0,1] dentro de cada sector
for sector in df['sector'].unique():
    mask = df['sector'] == sector
    z_min = df.loc[mask, 'z_score_local'].min()
    z_max = df.loc[mask, 'z_score_local'].max()
    rng   = z_max - z_min + 1e-9
    df.loc[mask, 'score_local'] = ((df.loc[mask, 'z_score_local'] - z_min) / rng).clip(0, 1)

# Semaforo local: siempre hay un VERDE LOCAL en cada sector
def semaforo_local(score_local, score_abs, kill_switch):
    if kill_switch:         return "ADVERSO"
    if score_local >= 0.75: return "VERDE_LOCAL"
    if score_local >= 0.45: return "AMARILLO_LOCAL"
    return "ROJO_LOCAL"

# Semaforo global: basado en score absoluto
def semaforo_global(score_abs, kill_switch):
    if kill_switch:              return "ADVERSO"
    if score_abs >= UMBRAL_VERDE: return "VERDE"
    if score_abs >= UMBRAL_AMARILLO: return "AMARILLO"
    return "ROJO"

df['semaforo_local']  = df.apply(lambda r: semaforo_local(r['score_local'], r['score_abs'], kill_switch_activo), axis=1)
df['semaforo_global'] = df.apply(lambda r: semaforo_global(r['score_abs'], kill_switch_activo), axis=1)

# ================================================================
# SELECCION TOP 5 TACTICO POR SECTOR
# Punto 1: maximo score local (nucleo del frente)
# Puntos 2-3: variantes del frente (min 500m de separacion)
# Puntos 4-5: trampas estaticas moduladas (mayor geometria costera)
# ================================================================
print("\nSeleccionando Top 5 tactico por sector...")

def seleccionar_top5(df_sector, sector_nombre, bonus_trampa_val):
    if len(df_sector) == 0:
        return pd.DataFrame()

    seleccionados = []
    df_ord = df_sector.sort_values('score_local', ascending=False).reset_index(drop=True)

    # Punto 1: nucleo del frente
    p1 = df_ord.iloc[0].copy()
    p1['rol'] = "NUCLEO_FRENTE"
    p1['rank_sector'] = 1
    seleccionados.append(p1)

    # Puntos 2 y 3: variantes con minimo 500m de separacion
    usados = [(float(p1['lat']), float(p1['lon']))]
    rank   = 2
    for _, row in df_ord.iterrows():
        if rank > 3:
            break
        dists_a_usados = [dist_km(float(row['lat']), float(row['lon']), u[0], u[1]) for u in usados]
        if min(dists_a_usados) >= 0.5:
            r = row.copy()
            r['rol'] = f"VARIANTE_{rank-1}"
            r['rank_sector'] = rank
            seleccionados.append(r)
            usados.append((float(row['lat']), float(row['lon'])))
            rank += 1

    # Puntos 4 y 5: trampas estaticas
    # Buscar puntos con alta geometria costera + bonus de surgencia activa
    # Aplicar bonus_trampa al score para que trampas suban cuando hay surgencia
    df_trampas = df_sector.copy()
    df_trampas['score_trampa'] = df_trampas['score_abs'] + bonus_trampa_val
    df_trampas['score_trampa'] = df_trampas['score_trampa'].clip(0, 1)
    df_trampas_ord = df_trampas.sort_values('score_trampa', ascending=False).reset_index(drop=True)

    rank_t = 4
    for _, row in df_trampas_ord.iterrows():
        if rank_t > 5:
            break
        dists_a_usados = [dist_km(float(row['lat']), float(row['lon']), u[0], u[1]) for u in usados]
        if min(dists_a_usados) >= 0.8:
            r = row.copy()
            r['rol'] = f"TRAMPA_{rank_t-3}"
            r['rank_sector'] = rank_t
            seleccionados.append(r)
            usados.append((float(row['lat']), float(row['lon'])))
            rank_t += 1

    return pd.DataFrame(seleccionados)

sectores_orden = ['COSTERO', 'SUR', 'NORTE', 'OESTE']
df_rep_list = []

for sector in sectores_orden:
    df_sec = df[df['sector'] == sector].copy()
    if len(df_sec) > 0:
        top5 = seleccionar_top5(df_sec, sector, bonus_trampa)
        print(f"  Sector {sector}: {len(df_sec)} puntos -> {len(top5)} seleccionados")
        df_rep_list.append(top5)
    else:
        print(f"  Sector {sector}: sin puntos")

if not df_rep_list:
    print("Sin resultados por sector -- pipeline termina")
    sys.exit(0)

df_rep = pd.concat(df_rep_list, ignore_index=True)
df_rep['confianza']        = CONFIANZA
df_rep['fecha']            = FECHA_HOY_STR
df_rep['hora_utc']         = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
df_rep['s1_dias']          = s1_ant if S1_DISPONIBLE else -1
df_rep['sst_ok']           = SST_DISPONIBLE
df_rep['s2_bloom_ok']      = S2_DISPONIBLE
df_rep['era5_ok']          = ERA5_DISPONIBLE
df_rep['corrientes_ok']    = CORRIENTES_DISPONIBLE
df_rep['oleaje_ok']        = OLEAJE_DISPONIBLE
df_rep['swh_medio']        = round(swh_medio, 2)
df_rep['kill_switch']      = kill_switch_activo
df_rep['sst_temp_medio']   = round(sst_temp_medio, 2) if sst_temp_medio else None
df_rep['indice_surgencia'] = round(indice_surgencia, 3)
df_rep['bonus_marea_global']    = round(bonus_marea_global, 3)
df_rep['marea_4am']  = VENTANAS_MAREA[4]['etiqueta']
df_rep['marea_5am']  = VENTANAS_MAREA[5]['etiqueta']
df_rep['marea_6am']  = VENTANAS_MAREA[6]['etiqueta']
df_rep['marea_7am']  = VENTANAS_MAREA[7]['etiqueta']
df_rep['marea_17pm'] = VENTANAS_MAREA[17]['etiqueta']
df_rep['marea_18pm'] = VENTANAS_MAREA[18]['etiqueta']
df_rep['uo_medio']         = round(uo_medio, 5)
df_rep['vo_medio']         = round(vo_medio, 5)
df_rep['bonus_trampa']     = round(bonus_trampa, 3)

print(f"\nTotal zonas reportadas: {len(df_rep)}")
sys.stdout.flush()

# ================================================================
# EXPORTAR REPORTE
# ================================================================
print("\nExportando reporte...")

try:
    ws = sh.worksheet('costero_reporte')
    ws.clear()
except:
    ws = sh.add_worksheet(title='costero_reporte', rows=200, cols=35)
set_with_dataframe(ws, df_rep)
print(f"Reporte exportado: {len(df_rep)} zonas en {df_rep['sector'].nunique()} sectores")
sys.stdout.flush()

# ================================================================
# HISTORIAL 90 DIAS
# ================================================================
print("\nActualizando historial...")

try:
    ws_hist = sh.worksheet('costero_historial')
    df_hist = pd.DataFrame(ws_hist.get_all_records())
except:
    ws_hist = sh.add_worksheet(title='costero_historial', rows=5000, cols=25)
    df_hist = pd.DataFrame()

cols_h = ['lat','lon','dist_km','sector','rol','rank_sector',
          'score_abs','score_local','semaforo_local','semaforo_global',
          'confianza','fecha','hora_utc','indice_surgencia','direccion']
df_h_nuevo    = df_rep[[c for c in cols_h if c in df_rep.columns]].copy()
df_hist_total = pd.concat([df_hist, df_h_nuevo], ignore_index=True) if len(df_hist) > 0 else df_h_nuevo
fecha_limite  = (datetime.utcnow() - timedelta(days=HISTORIAL_DIAS)).strftime("%Y-%m-%d")
df_hist_total = df_hist_total[df_hist_total['fecha'] >= fecha_limite].reset_index(drop=True)
ws_hist.clear()
set_with_dataframe(ws_hist, df_hist_total)
print(f"Historial: {len(df_hist_total)} registros")
sys.stdout.flush()

# ================================================================
# IPO COSTERO diferenciado por sector
# ================================================================
print("\nCalculando IPO Costero por sector...")

ipo_rows = []
fecha_3d = (datetime.utcnow() - timedelta(days=IPO_DIAS_PELAGICOS)).strftime("%Y-%m-%d")
fecha_5d = (datetime.utcnow() - timedelta(days=IPO_DIAS_DEMERSALES)).strftime("%Y-%m-%d")

for _, zona in df_rep.iterrows():
    lat_z = float(zona['lat'])
    lon_z = float(zona['lon'])
    if len(df_hist_total) > 0:
        df_hist_total['dist_zona'] = df_hist_total.apply(
            lambda r: dist_km(lat_z, lon_z, float(r['lat']), float(r['lon'])), axis=1)
        vecinos = df_hist_total[df_hist_total['dist_zona'] <= 2.0]
    else:
        vecinos = pd.DataFrame()

    vec_3d  = vecinos[vecinos['fecha'] >= fecha_3d] if len(vecinos) > 0 else pd.DataFrame()
    n_3d    = len(vec_3d['fecha'].unique()) if len(vec_3d) > 0 else 0
    ipo_pel = min(n_3d / IPO_DIAS_PELAGICOS, 1.0)
    vec_5d  = vecinos[vecinos['fecha'] >= fecha_5d] if len(vecinos) > 0 else pd.DataFrame()
    n_5d    = len(vec_5d['fecha'].unique()) if len(vec_5d) > 0 else 0
    ipo_dem = min(n_5d / IPO_DIAS_DEMERSALES, 1.0)

    def ipo_label(val):
        if val >= 0.75:   return "CONFIRMADA"
        elif val >= 0.50: return "EN OBSERVACION"
        else:             return "INESTABLE"

    ipo_rows.append({
        'lat': lat_z, 'lon': lon_z,
        'dist_km': float(zona['dist_km']),
        'sector': zona['sector'],
        'rol': zona['rol'],
        'rank_sector': int(zona['rank_sector']),
        'score_abs': float(zona['score_abs']),
        'score_local': round(float(zona['score_local']), 4),
        'semaforo_local': zona['semaforo_local'],
        'lat_T16': float(zona['lat_T16']),
        'lon_T16': float(zona['lon_T16']),
        'desp_km': float(zona['desp_km']),
        'direccion': zona['direccion'],
        'ipo_pelagico': round(ipo_pel, 3),
        'ipo_demersal': round(ipo_dem, 3),
        'label_pel': ipo_label(ipo_pel),
        'label_dem': ipo_label(ipo_dem),
        'fecha': FECHA_HOY_STR
    })

df_ipo = pd.DataFrame(ipo_rows)
try:
    ws_ipo = sh.worksheet('costero_ipo')
    ws_ipo.clear()
except:
    ws_ipo = sh.add_worksheet(title='costero_ipo', rows=200, cols=20)
set_with_dataframe(ws_ipo, df_ipo)
print(f"IPO: {len(df_ipo)} zonas")
sys.stdout.flush()

# ================================================================
# RESUMEN FINAL
# ================================================================
print("\n" + "=" * 60)
print(f"PredictaMAR Costero v1.2 COMPLETO -- {FECHA_HOY_STR}")
print(f"Confianza: {CONFIANZA}")
print(f"SST L4:      {'OK T=' + str(round(sst_temp_medio,1)) if SST_DISPONIBLE else 'NO'}")
print(f"ERA5:        {'OK ind=' + str(round(indice_surgencia,3)) if ERA5_DISPONIBLE else 'NO'}")
print(f"S2 bloom:    {'OK Fc=' + str(round(Fc_s2,2)) if S2_DISPONIBLE else 'NO (nubosidad)'}")
print(f"S1 SAR:      {'OK Fc=' + str(round(Fc_s1,2)) if S1_DISPONIBLE else 'NO'}")
print(f"ALOS-2:      {'OK Fc=' + str(round(Fc_alos2,2)) if ALOS2_DISPONIBLE else 'NO'}")
print(f"Oleaje:      {'ADVERSO SWH=' + str(round(swh_medio,2)) if kill_switch_activo else 'OK SWH=' + str(round(swh_medio,2)) + 'm'}")
print(f"Bonus trampa: +{bonus_trampa:.3f}")
print(f"Sectores: {df_rep['sector'].value_counts().to_dict()}")
print(f"Total zonas: {len(df_rep)}")
print(f"Fin: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()
