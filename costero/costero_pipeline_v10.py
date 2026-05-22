0 · PY
# ================================================================
# PredictaMAR Costero v1.0 — PIPELINE AUTOMATICO
# Puerto Chorrillos · 0-20 km · Sistema Corriente de Humboldt
# Corre desde GitHub Actions: 3:00 AM y 3:00 PM Lima (UTC-5)
# ================================================================
 
import sys
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import requests
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime, timedelta
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
import copernicusmarine
import ee
import warnings
warnings.filterwarnings('ignore')
 
print("=" * 60)
print("PredictaMAR Costero v1.0 — PIPELINE AUTOMATICO")
print(f"Inicio: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()
 
# ── Importar configuración ────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from costero_config import *
 
# ── Credenciales ──────────────────────────────────────────────────
CMEMS_USER = os.environ["CMEMS_USER"]
CMEMS_PASS = os.environ["CMEMS_PASS"]
SHEET_ID   = os.environ["SHEET_ID"]
GOOGLE_SA  = os.environ["GOOGLE_SA_JSON"]
 
print("Credenciales cargadas OK")
sys.stdout.flush()
 
# ── Google Sheets ─────────────────────────────────────────────────
sa_info  = json.loads(GOOGLE_SA)
scopes   = ["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]
sa_creds = SACredentials.from_service_account_info(sa_info, scopes=scopes)
gc       = gspread.authorize(sa_creds)
sh       = gc.open_by_key(SHEET_ID)
print(f"Sheets OK: {sh.title}")
sys.stdout.flush()
 
# ── CMEMS login ───────────────────────────────────────────────────
copernicusmarine.login(
    username=CMEMS_USER,
    password=CMEMS_PASS,
    force_overwrite=True
)
print("CMEMS OK")
sys.stdout.flush()
 
# ── GEE init ─────────────────────────────────────────────────────
gee_creds = ee.ServiceAccountCredentials(
    sa_info["client_email"],
    key_data=sa_info["private_key"]
)
ee.Initialize(gee_creds)
print("GEE OK")
sys.stdout.flush()
 
# ── Parámetros de fecha ───────────────────────────────────────────
FECHA_HOY     = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
FECHA_HOY_STR = FECHA_HOY.strftime("%Y-%m-%d")
MES_ACTUAL    = FECHA_HOY.month
DRIVE_BASE    = "/tmp/predictamar_costero"
os.makedirs(f"{DRIVE_BASE}/raw", exist_ok=True)
 
print(f"Fecha: {FECHA_HOY_STR} | Mes: {MES_ACTUAL}")
sys.stdout.flush()
 
# ── Geometría GEE ────────────────────────────────────────────────
AOI = ee.Geometry.Rectangle([LON_MIN_C, LAT_MIN_C, LON_MAX_C, LAT_MAX_C])
PUNTO_CHORRILLOS = ee.Geometry.Point([LON_CHORRILLOS, LAT_CHORRILLOS])
 
# ================================================================
# BLOQUE 1 — SENTINEL-1 SAR (banda C · 10m · todo clima)
# ================================================================
print("\n[1/5] Descargando Sentinel-1 SAR...")
sys.stdout.flush()
 
S1_DISPONIBLE = False
s1_antiguedad_dias = 99
 
try:
    fecha_s1_inicio = (FECHA_HOY - timedelta(days=FC_S1_MAX_DIAS + 2)).strftime("%Y-%m-%d")
 
    s1_col = (ee.ImageCollection(GEE_S1)
                .filterBounds(AOI)
                .filterDate(fecha_s1_inicio, FECHA_HOY_STR)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select(['VV']))
 
    s1_count = s1_col.size().getInfo()
 
    if s1_count > 0:
        s1_imagen = s1_col.sort('system:time_start', False).first()
        s1_fecha  = datetime.fromtimestamp(
            s1_imagen.get('system:time_start').getInfo() / 1000
        )
        s1_antiguedad_dias = (FECHA_HOY - s1_fecha).days
 
        # Calcular gradiente de rugosidad (proxy de frentes)
        s1_smooth  = s1_imagen.focal_mean(radius=2, kernelType='circle', units='pixels')
        s1_grad    = s1_smooth.gradient().abs().reduce(ee.Reducer.max())
        s1_norm    = s1_grad.unitScale(
            s1_grad.reduceRegion(ee.Reducer.percentile([5]), AOI, 100).values().get(0),
            s1_grad.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('s1_rugosidad')
 
        S1_DISPONIBLE = True
        Fc_s1 = max(0, 1 - s1_antiguedad_dias / FC_S1_MAX_DIAS)
        print(f"  S1 OK — antigüedad: {s1_antiguedad_dias}d | Fc: {Fc_s1:.2f}")
    else:
        print("  S1 sin dato reciente — usando climatológico")
 
except Exception as e:
    print(f"  S1 error: {e}")
 
sys.stdout.flush()
 
# ================================================================
# BLOQUE 2 — SENTINEL-2 ÓPTICO (10m · solo cielo despejado)
# ================================================================
print("\n[2/5] Descargando Sentinel-2 óptico...")
sys.stdout.flush()
 
S2_DISPONIBLE  = False
s2_antiguedad_dias = 99
 
try:
    fecha_s2_inicio = (FECHA_HOY - timedelta(days=FC_S2_MAX_DIAS + 3)).strftime("%Y-%m-%d")
 
    def mask_s2_nubes(image):
        scl  = image.select('SCL')
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)
 
    s2_col = (ee.ImageCollection(GEE_S2)
                .filterBounds(AOI)
                .filterDate(fecha_s2_inicio, FECHA_HOY_STR)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .select(['B3', 'B4', 'B8', 'SCL'])
                .map(mask_s2_nubes))
 
    s2_count = s2_col.size().getInfo()
 
    if s2_count > 0:
        s2_imagen = s2_col.sort('system:time_start', False).first()
        s2_fecha  = datetime.fromtimestamp(
            s2_imagen.get('system:time_start').getInfo() / 1000
        )
        s2_antiguedad_dias = (FECHA_HOY - s2_fecha).days
 
        # Índice de turbidez costera (ratio B4/B3)
        b3 = s2_imagen.select('B3').toFloat().divide(10000)
        b4 = s2_imagen.select('B4').toFloat().divide(10000)
        b8 = s2_imagen.select('B8').toFloat().divide(10000)
 
        turbidez = b4.divide(b3.add(1e-6)).rename('turbidez')
 
        # Clorofila proxy lag: buscamos dato de hace LAG_CHL_DIAS días
        fecha_lag_inicio = (FECHA_HOY - timedelta(days=LAG_CHL_DIAS + 3)).strftime("%Y-%m-%d")
        fecha_lag_fin    = (FECHA_HOY - timedelta(days=LAG_CHL_DIAS)).strftime("%Y-%m-%d")
 
        s2_lag = (ee.ImageCollection(GEE_S2)
                    .filterBounds(AOI)
                    .filterDate(fecha_lag_inicio, fecha_lag_fin)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .select(['B3', 'B4', 'SCL'])
                    .map(mask_s2_nubes)
                    .median())
 
        b3_lag  = s2_lag.select('B3').toFloat().divide(10000)
        b4_lag  = s2_lag.select('B4').toFloat().divide(10000)
        chl_lag = b3_lag.divide(b4_lag.add(1e-6)).rename('chl_lag')
 
        S2_DISPONIBLE  = True
        Fc_s2 = max(0, 1 - s2_antiguedad_dias / FC_S2_MAX_DIAS)
        print(f"  S2 OK — antigüedad: {s2_antiguedad_dias}d | Fc: {Fc_s2:.2f} | lag CHL {LAG_CHL_DIAS}d OK")
    else:
        print(f"  S2 sin dato limpio en últimos {FC_S2_MAX_DIAS + 3} días — nubosidad")
 
except Exception as e:
    print(f"  S2 error: {e}")
 
sys.stdout.flush()
 
# ================================================================
# BLOQUE 3 — ALOS-2 PALSAR-2 (banda L · 25m · todo clima)
# ================================================================
print("\n[3/5] Descargando ALOS-2 PALSAR-2...")
sys.stdout.flush()
 
ALOS2_DISPONIBLE = False
alos2_antiguedad_dias = 99
 
try:
    fecha_alos_inicio = (FECHA_HOY - timedelta(days=FC_ALOS2_MAX_DIAS + 5)).strftime("%Y-%m-%d")
 
    alos2_col = (ee.ImageCollection(GEE_ALOS2)
                   .filterBounds(AOI)
                   .filterDate(fecha_alos_inicio, FECHA_HOY_STR)
                   .select(['HH']))
 
    alos2_count = alos2_col.size().getInfo()
 
    if alos2_count > 0:
        alos2_imagen = alos2_col.sort('system:time_start', False).first()
        alos2_fecha  = datetime.fromtimestamp(
            alos2_imagen.get('system:time_start').getInfo() / 1000
        )
        alos2_antiguedad_dias = (FECHA_HOY - alos2_fecha).days
 
        # Gradiente estructural banda L (corrientes y batimetría)
        alos2_smooth = alos2_imagen.focal_mean(radius=3, kernelType='circle', units='pixels')
        alos2_grad   = alos2_smooth.gradient().abs().reduce(ee.Reducer.max())
        alos2_norm   = alos2_grad.unitScale(
            alos2_grad.reduceRegion(ee.Reducer.percentile([5]), AOI, 100).values().get(0),
            alos2_grad.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('alos2_estructura')
 
        ALOS2_DISPONIBLE = True
        Fc_alos2 = max(0, 1 - alos2_antiguedad_dias / FC_ALOS2_MAX_DIAS)
        print(f"  ALOS-2 OK — antigüedad: {alos2_antiguedad_dias}d | Fc: {Fc_alos2:.2f}")
    else:
        print("  ALOS-2 sin dato reciente")
 
except Exception as e:
    print(f"  ALOS-2 error: {e}")
 
sys.stdout.flush()
 
# ================================================================
# BLOQUE 4 — SST L4 CMEMS (sin nubes · diario)
# ================================================================
print("\n[4/5] Descargando SST L4 CMEMS...")
sys.stdout.flush()
 
SST_DISPONIBLE = False
 
try:
    sst_path = f"{DRIVE_BASE}/raw/sst_l4_costero.nc"
    if os.path.exists(sst_path):
        os.remove(sst_path)
 
    copernicusmarine.subset(
        dataset_id           = CMEMS_SST_L4,
        variables            = ["analysed_sst"],
        minimum_latitude     = LAT_MIN_C,
        maximum_latitude     = LAT_MAX_C,
        minimum_longitude    = LON_MIN_C,
        maximum_longitude    = LON_MAX_C,
        start_datetime       = (FECHA_HOY - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime         = FECHA_HOY.strftime("%Y-%m-%dT23:59:59"),
        output_filename      = "sst_l4_costero.nc",
        output_directory     = f"{DRIVE_BASE}/raw",
        username             = CMEMS_USER,
        password             = CMEMS_PASS,
        disable_progress_bar = True
    )
 
    ds_sst   = xr.open_dataset(sst_path)
    sst_data = ds_sst['analysed_sst'].values
    sst_data = sst_data - 273.15  # Kelvin a Celsius
    sst_mean = np.nanmean(sst_data, axis=0) if sst_data.ndim == 3 else sst_data
 
    SST_DISPONIBLE = True
    print(f"  SST L4 OK — rango: {np.nanmin(sst_mean):.1f}°C a {np.nanmax(sst_mean):.1f}°C")
 
except Exception as e:
    print(f"  SST L4 error: {e}")
 
sys.stdout.flush()
 
# ================================================================
# BLOQUE 5 — BATIMETRÍA GEBCO (estática)
# ================================================================
print("\n[5/5] Cargando batimetría GEBCO...")
sys.stdout.flush()
 
try:
    gebco = ee.Image(GEE_GEBCO).select('elevation').clip(AOI)
    # Pendiente del fondo (slope) — proxy de micro-surgencia topográfica
    gebco_slope = ee.Terrain.slope(gebco).rename('slope')
    print("  GEBCO OK")
except Exception as e:
    print(f"  GEBCO error: {e}")
 
sys.stdout.flush()
 
# ================================================================
# CÁLCULO DEL MICROSCORE
# ================================================================
print("\nCalculando MicroScore Costero...")
sys.stdout.flush()
 
# Determinar modo de operación
if S2_DISPONIBLE and s2_antiguedad_dias <= FC_S2_MAX_DIAS:
    MODO       = "NORMAL"
    W_ACTIVOS  = W_NORMAL.copy()
    CONFIANZA_BASE = CONFIANZA_ALTA
else:
    MODO       = "DEGRADADO"
    W_ACTIVOS  = W_DEGRADADO.copy()
    CONFIANZA_BASE = CONFIANZA_MEDIA
    print(f"  Modo degradado — S2 con {s2_antiguedad_dias}d de antigüedad")
 
if not S1_DISPONIBLE:
    CONFIANZA_BASE = CONFIANZA_BAJA
    print("  Confianza baja — S1 no disponible, usando climatológico")
 
print(f"  Modo: {MODO} | Confianza base: {CONFIANZA_BASE}")
 
# Generar grilla de puntos sobre el AOI (cada ~500m)
lats = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0045)
lons = np.arange(LON_MIN_C, LON_MAX_C, 0.0045)
grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')
puntos_flat = [(float(la), float(lo))
               for la, lo in zip(grid_lats.ravel(), grid_lons.ravel())]
 
# Calcular distancia desde Chorrillos para cada punto
def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(dlat**2 + dlon**2)
 
# Extraer scores por punto usando GEE
print(f"  Evaluando {len(puntos_flat)} puntos en la grilla...")
sys.stdout.flush()
 
resultados = []
batch_size = 100
 
for i in range(0, min(len(puntos_flat), 500), batch_size):
    batch = puntos_flat[i:i + batch_size]
    features = []
 
    for lat, lon in batch:
        dist = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, lat, lon)
        if dist > RADIO_MAX_KM or dist < RADIO_ORILLA_KM:
            continue
        features.append(ee.Feature(
            ee.Geometry.Point([lon, lat]),
            {'lat': lat, 'lon': lon, 'dist_km': dist}
        ))
 
    if not features:
        continue
 
    try:
        fc = ee.FeatureCollection(features)
 
        # Capas a evaluar
        capas = []
        if S1_DISPONIBLE:
            capas.append(s1_norm.multiply(W_ACTIVOS["s1_sar"] * Fc_s1))
        if S2_DISPONIBLE and MODO == "NORMAL":
            capas.append(turbidez.unitScale(0, 2).multiply(W_ACTIVOS["s2_optico"] * Fc_s2))
            capas.append(chl_lag.unitScale(0, 3).multiply(0.1))
        capas.append(gebco_slope.unitScale(0, 30).multiply(W_ACTIVOS["gebco"]))
        if ALOS2_DISPONIBLE:
            capas.append(alos2_norm.multiply(W_ACTIVOS["alos2"] * Fc_alos2))
 
        if not capas:
            continue
 
        score_imagen = ee.ImageCollection(capas).sum().rename('score')
 
        res = score_imagen.reduceRegions(
            collection = fc,
            reducer    = ee.Reducer.mean(),
            scale      = 30
        ).getInfo()
 
        for feat in res['features']:
            p = feat['properties']
            if p.get('mean') is not None:
                resultados.append({
                    'lat':     p['lat'],
                    'lon':     p['lon'],
                    'dist_km': round(p['dist_km'], 1),
                    'score':   round(float(p['mean']), 4)
                })
 
    except Exception as e:
        print(f"  Batch {i} error: {e}")
        continue
 
print(f"  Puntos con score: {len(resultados)}")
sys.stdout.flush()
 
# ================================================================
# EXPORTAR REPORTE
# ================================================================
print("\nExportando reporte...")
sys.stdout.flush()
 
if not resultados:
    print("Sin resultados — pipeline termina sin exportar")
    sys.exit(0)
 
df = pd.DataFrame(resultados).sort_values('score', ascending=False).reset_index(drop=True)
 
def get_semaforo(score):
    if score >= UMBRAL_VERDE:    return "VERDE"
    elif score >= UMBRAL_AMARILLO: return "AMARILLO"
    else:                          return "ROJO"
 
def get_confianza(modo, s1_ok, s2_ok):
    if modo == "NORMAL" and s1_ok and s2_ok: return "ALTA"
    elif s1_ok:                               return "MEDIA"
    else:                                     return "BAJA"
 
# Top zonas por anillo de 5 km
anillos = [(0,5),(5,10),(10,15),(15,20)]
df_reporte = []
 
for d_min, d_max in anillos:
    anillo = df[(df['dist_km'] >= d_min) & (df['dist_km'] < d_max)].head(6)
    df_reporte.append(anillo)
    print(f"  Anillo {d_min}-{d_max}km: {len(anillo)} zonas")
 
df_reporte = pd.concat(df_reporte, ignore_index=True)
df_reporte['semaforo']   = df_reporte['score'].apply(get_semaforo)
df_reporte['confianza']  = get_confianza(MODO, S1_DISPONIBLE, S2_DISPONIBLE)
df_reporte['modo']       = MODO
df_reporte['fecha']      = FECHA_HOY_STR
df_reporte['hora_utc']   = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
df_reporte['s1_dias']    = s1_antiguedad_dias if S1_DISPONIBLE else -1
df_reporte['s2_dias']    = s2_antiguedad_dias if S2_DISPONIBLE else -1
df_reporte['alos2_dias'] = alos2_antiguedad_dias if ALOS2_DISPONIBLE else -1
df_reporte['sst_ok']     = SST_DISPONIBLE
 
# Exportar reporte diario
try:
    ws = sh.worksheet('costero_reporte')
    ws.clear()
except:
    ws = sh.add_worksheet(title='costero_reporte', rows=200, cols=20)
 
set_with_dataframe(ws, df_reporte)
print(f"Reporte exportado: {len(df_reporte)} zonas")
sys.stdout.flush()
 
# ================================================================
# HISTORIAL 90 DÍAS
# ================================================================
print("\nActualizando historial 90 días...")
sys.stdout.flush()
 
try:
    ws_hist = sh.worksheet('costero_historial')
    df_hist = pd.DataFrame(ws_hist.get_all_records())
except:
    ws_hist = sh.add_worksheet(title='costero_historial', rows=5000, cols=20)
    df_hist = pd.DataFrame()
 
df_hist_nuevo = df_reporte[['lat','lon','dist_km','score','semaforo',
                              'confianza','modo','fecha','hora_utc',
                              's1_dias','s2_dias','alos2_dias']].copy()
 
if len(df_hist) > 0:
    df_hist_total = pd.concat([df_hist, df_hist_nuevo], ignore_index=True)
else:
    df_hist_total = df_hist_nuevo
 
# Conservar solo 90 días
fecha_limite = (datetime.utcnow() - timedelta(days=HISTORIAL_DIAS)).strftime("%Y-%m-%d")
df_hist_total = df_hist_total[df_hist_total['fecha'] >= fecha_limite].reset_index(drop=True)
 
ws_hist.clear()
set_with_dataframe(ws_hist, df_hist_total)
print(f"Historial actualizado: {len(df_hist_total)} registros")
sys.stdout.flush()
 
# ================================================================
# IPO COSTERO diferenciado
# ================================================================
print("\nCalculando IPO Costero...")
sys.stdout.flush()
 
ipo_rows = []
fecha_3d = (datetime.utcnow() - timedelta(days=IPO_DIAS_PELAGICOS)).strftime("%Y-%m-%d")
fecha_5d = (datetime.utcnow() - timedelta(days=IPO_DIAS_DEMERSALES)).strftime("%Y-%m-%d")
 
for _, zona in df_reporte.iterrows():
    lat_z = float(zona['lat'])
    lon_z = float(zona['lon'])
 
    if len(df_hist_total) > 0:
        df_hist_total['dist_zona'] = df_hist_total.apply(
            lambda r: dist_km(lat_z, lon_z,
                              float(r['lat']), float(r['lon'])), axis=1
        )
        vecinos = df_hist_total[df_hist_total['dist_zona'] <= 2.0]
    else:
        vecinos = pd.DataFrame()
 
    # IPO pelágicos (3 días)
    vec_3d   = vecinos[vecinos['fecha'] >= fecha_3d] if len(vecinos) > 0 else pd.DataFrame()
    n_3d     = len(vec_3d['fecha'].unique()) if len(vec_3d) > 0 else 0
    ipo_pel  = min(n_3d / IPO_DIAS_PELAGICOS, 1.0)
 
    # IPO demersales (5 días)
    vec_5d   = vecinos[vecinos['fecha'] >= fecha_5d] if len(vecinos) > 0 else pd.DataFrame()
    n_5d     = len(vec_5d['fecha'].unique()) if len(vec_5d) > 0 else 0
    ipo_dem  = min(n_5d / IPO_DIAS_DEMERSALES, 1.0)
 
    def ipo_label(val):
        if val >= 0.75:   return "CONFIRMADA"
        elif val >= 0.50: return "EN OBSERVACION"
        else:             return "INESTABLE"
 
    ipo_rows.append({
        'lat':          lat_z,
        'lon':          lon_z,
        'dist_km':      float(zona['dist_km']),
        'score':        float(zona['score']),
        'ipo_pelagico': round(ipo_pel, 3),
        'ipo_demersal': round(ipo_dem, 3),
        'label_pel':    ipo_label(ipo_pel),
        'label_dem':    ipo_label(ipo_dem),
        'n_dias_3d':    n_3d,
        'n_dias_5d':    n_5d,
        'fecha':        FECHA_HOY_STR
    })
 
df_ipo = pd.DataFrame(ipo_rows)
 
try:
    ws_ipo = sh.worksheet('costero_ipo')
    ws_ipo.clear()
except:
    ws_ipo = sh.add_worksheet(title='costero_ipo', rows=200, cols=15)
 
set_with_dataframe(ws_ipo, df_ipo)
print(f"IPO calculado: {len(df_ipo)} zonas")
sys.stdout.flush()
 
# ================================================================
# RESUMEN FINAL
# ================================================================
print("\n" + "=" * 60)
print(f"PredictaMAR Costero v1.0 COMPLETO — {FECHA_HOY_STR}")
print(f"Modo: {MODO} | Confianza: {get_confianza(MODO, S1_DISPONIBLE, S2_DISPONIBLE)}")
print(f"S1: {'OK ' + str(s1_antiguedad_dias) + 'd' if S1_DISPONIBLE else 'NO'} | "
      f"S2: {'OK ' + str(s2_antiguedad_dias) + 'd' if S2_DISPONIBLE else 'NO'} | "
      f"ALOS2: {'OK ' + str(alos2_antiguedad_dias) + 'd' if ALOS2_DISPONIBLE else 'NO'} | "
      f"SST: {'OK' if SST_DISPONIBLE else 'NO'}")
print(f"Zonas reportadas: {len(df_reporte)}")
print(f"Historial total: {len(df_hist_total)} registros")
print(f"Fin: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()
