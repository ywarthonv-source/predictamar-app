# ================================================================
# PredictaMAR Costero v1.1 -- PIPELINE CADENA TROFICA TEMPORAL
# Puerto Chorrillos - 0-20 km - Sistema Corriente de Humboldt
# Logica: surgencia (T-7d) -> bloom (T-4d) -> cardumen (hoy)
# Corre desde GitHub Actions: 3:00 AM y 3:00 PM Lima (UTC-5)
# ================================================================

import sys
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime, timedelta
import copernicusmarine
import ee
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PredictaMAR Costero v1.1 -- CADENA TROFICA TEMPORAL")
print(f"Inicio: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from costero_config import *

# -- Credenciales
CMEMS_USER = os.environ["CMEMS_USER"]
CMEMS_PASS = os.environ["CMEMS_PASS"]
SHEET_ID   = os.environ["COSTERO_SHEET_ID"]
GOOGLE_SA  = os.environ["GOOGLE_SA_JSON"]
print("Credenciales OK")
sys.stdout.flush()

# -- Google Sheets
sa_info  = json.loads(GOOGLE_SA)
scopes   = ["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]
sa_creds = SACredentials.from_service_account_info(sa_info, scopes=scopes)
gc       = gspread.authorize(sa_creds)
sh       = gc.open_by_key(SHEET_ID)
print(f"Sheets OK: {sh.title}")
sys.stdout.flush()

# -- CMEMS login
copernicusmarine.login(
    username=CMEMS_USER,
    password=CMEMS_PASS,
    force_overwrite=True
)
print("CMEMS OK")
sys.stdout.flush()

# -- GEE init
gee_creds = ee.ServiceAccountCredentials(
    sa_info["client_email"],
    key_data=sa_info["private_key"]
)
ee.Initialize(gee_creds)
print("GEE OK")
sys.stdout.flush()

# -- Fechas clave de la cadena trofica
FECHA_HOY      = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
FECHA_HOY_STR  = FECHA_HOY.strftime("%Y-%m-%d")
FECHA_T4       = FECHA_HOY - timedelta(days=LAG_CHL_DIAS)       # bloom
FECHA_T7       = FECHA_HOY - timedelta(days=7)                  # surgencia
DRIVE_BASE     = "/tmp/predictamar_costero"
os.makedirs(f"{DRIVE_BASE}/raw", exist_ok=True)

print(f"Fecha hoy:      {FECHA_HOY_STR}")
print(f"Fecha bloom:    {FECHA_T4.strftime('%Y-%m-%d')} (T-{LAG_CHL_DIAS}d)")
print(f"Fecha surgencia:{FECHA_T7.strftime('%Y-%m-%d')} (T-7d)")
sys.stdout.flush()

# -- Geometria GEE
AOI = ee.Geometry.Rectangle([LON_MIN_C, LAT_MIN_C, LON_MAX_C, LAT_MAX_C])

# ================================================================
# ESLABON 1 -- SURGENCIA (T-7 dias)
# SST L4 CMEMS -- temperatura sin nubes -- detecta donde subio
# agua fria desde el fondo hace 7 dias
# ================================================================
print("\n[ESLABON 1/4] Surgencia T-7d -- SST L4 CMEMS...")
sys.stdout.flush()

SST_DISPONIBLE   = False
sst_grad_imagen  = None
sst_valor_medio  = None

try:
    sst_path = f"{DRIVE_BASE}/raw/sst_l4.nc"
    if os.path.exists(sst_path):
        os.remove(sst_path)

    copernicusmarine.subset(
        dataset_id           = CMEMS_SST_L4,
        variables            = ["analysed_sst"],
        minimum_latitude     = LAT_MIN_C,
        maximum_latitude     = LAT_MAX_C,
        minimum_longitude    = LON_MIN_C,
        maximum_longitude    = LON_MAX_C,
        start_datetime       = (FECHA_T7 - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime         = FECHA_T7.strftime("%Y-%m-%dT23:59:59"),
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

    # Calcular gradiente termico -- indica donde hubo surgencia
    gy, gx    = np.gradient(np.where(np.isnan(sst_mean), 0, sst_mean))
    sst_grad  = np.sqrt(gx**2 + gy**2)
    sst_grad[np.isnan(sst_mean)] = np.nan

    # Normalizar
    p5  = np.nanpercentile(sst_grad, 5)
    p95 = np.nanpercentile(sst_grad, 95)
    sst_grad_norm = np.clip((sst_grad - p5) / (p95 - p5 + 1e-9), 0, 1)

    sst_valor_medio = float(np.nanmean(sst_mean))

    # Convertir a imagen GEE para reduceRegions
    lat_sst = ds_sst['latitude'].values if 'latitude' in ds_sst else ds_sst['lat'].values
    lon_sst = ds_sst['longitude'].values if 'longitude' in ds_sst else ds_sst['lon'].values

    SST_DISPONIBLE = True
    print(f"  SST OK -- T medio: {sst_valor_medio:.1f}C | gradiente calculado")

except Exception as e:
    print(f"  SST L4 error: {e}")
sys.stdout.flush()

# ================================================================
# ESLABON 2 -- BLOOM (T-4 dias)
# Sentinel-2 si hay cielo despejado, o ERA5 viento como proxy
# ================================================================
print("\n[ESLABON 2/4] Bloom T-4d -- S2 optico / ERA5 viento...")
sys.stdout.flush()

S2_BLOOM_DISPONIBLE = False
ERA5_DISPONIBLE     = False
chl_bloom_imagen    = None
viento_surgencia    = None
Fc_s2_bloom         = 0.0

# Intentar S2 en la ventana del bloom
try:
    fecha_bloom_ini = (FECHA_T4 - timedelta(days=3)).strftime("%Y-%m-%d")
    fecha_bloom_fin = (FECHA_T4 + timedelta(days=2)).strftime("%Y-%m-%d")

    def mask_s2_nubes(image):
        scl  = image.select('SCL')
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)

    s2_bloom_col = (ee.ImageCollection(GEE_S2)
                      .filterBounds(AOI)
                      .filterDate(fecha_bloom_ini, fecha_bloom_fin)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .select(['B3', 'B4', 'SCL'])
                      .map(mask_s2_nubes))

    if s2_bloom_col.size().getInfo() > 0:
        s2_bloom  = s2_bloom_col.sort('system:time_start', False).first()
        s2_fecha  = datetime.fromtimestamp(
            s2_bloom.get('system:time_start').getInfo() / 1000)
        dias_diff = abs((FECHA_T4 - s2_fecha).days)

        b3 = s2_bloom.select('B3').toFloat().divide(10000)
        b4 = s2_bloom.select('B4').toFloat().divide(10000)
        # Indice de bloom: agua verde = alta clorofila
        chl_bloom_imagen = b3.divide(b4.add(1e-6)).rename('chl_bloom')

        S2_BLOOM_DISPONIBLE = True
        Fc_s2_bloom = max(0, 1 - dias_diff / 5)
        print(f"  S2 bloom OK -- fecha: {s2_fecha.strftime('%Y-%m-%d')} | Fc: {Fc_s2_bloom:.2f}")
    else:
        print(f"  S2 bloom sin dato limpio -- usando ERA5 viento")

except Exception as e:
    print(f"  S2 bloom error: {e}")

# ERA5 viento como proxy de surgencia activa
try:
    era5_path = f"{DRIVE_BASE}/raw/era5_viento.nc"
    if os.path.exists(era5_path):
        os.remove(era5_path)

    copernicusmarine.subset(
        dataset_id           = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H-i",
        variables            = ["eastward_wind", "northward_wind"],
        minimum_latitude     = LAT_MIN_C - 0.5,
        maximum_latitude     = LAT_MAX_C + 0.5,
        minimum_longitude    = LON_MIN_C - 0.5,
        maximum_longitude    = LON_MAX_C + 0.5,
        start_datetime       = FECHA_T7.strftime("%Y-%m-%dT00:00:00"),
        end_datetime         = FECHA_HOY.strftime("%Y-%m-%dT23:59:59"),
        output_filename      = "era5_viento.nc",
        output_directory     = f"{DRIVE_BASE}/raw",
        username             = CMEMS_USER,
        password             = CMEMS_PASS,
        disable_progress_bar = True
    )

    ds_viento = xr.open_dataset(era5_path)
    u = ds_viento['eastward_wind'].values
    v = ds_viento['northward_wind'].values

    # Componente paralela a la costa peruana (N-S)
    # Viento del sur (v > 0 hacia norte) activa surgencia costera
    v_mean = float(np.nanmean(v))
    u_mean = float(np.nanmean(u))
    # Indice de surgencia: componente sur positiva = surgencia activa
    indice_surgencia = max(0, v_mean) / (abs(v_mean) + abs(u_mean) + 1e-9)

    ERA5_DISPONIBLE  = True
    viento_surgencia = indice_surgencia
    print(f"  ERA5 OK -- viento U: {u_mean:.2f} V: {v_mean:.2f} | indice surgencia: {indice_surgencia:.2f}")

except Exception as e:
    print(f"  ERA5 viento error: {e}")
sys.stdout.flush()

# ================================================================
# ESLABON 3 -- CONDICIONES ACTUALES (hoy)
# SAR rugosidad superficial + batimetria
# ================================================================
print("\n[ESLABON 3/4] Condiciones actuales -- SAR + batimetria...")
sys.stdout.flush()

S1_DISPONIBLE      = False
s1_antiguedad_dias = 99
Fc_s1              = 0.0
s1_norm            = None
ALOS2_DISPONIBLE   = False
alos2_norm         = None
Fc_alos2           = 0.0
gebco_slope        = None

# Sentinel-1 SAR
try:
    fecha_s1_ini = (FECHA_HOY - timedelta(days=FC_S1_MAX_DIAS + 2)).strftime("%Y-%m-%d")
    s1_col = (ee.ImageCollection(GEE_S1)
                .filterBounds(AOI)
                .filterDate(fecha_s1_ini, FECHA_HOY_STR)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select(['VV']))

    if s1_col.size().getInfo() > 0:
        s1_img  = s1_col.sort('system:time_start', False).first()
        s1_ts   = datetime.fromtimestamp(
            s1_img.get('system:time_start').getInfo() / 1000)
        s1_antiguedad_dias = (FECHA_HOY - s1_ts).days
        s1_smooth = s1_img.focal_mean(radius=2, kernelType='circle', units='pixels')
        s1_grad   = s1_smooth.gradient().abs().reduce(ee.Reducer.max())
        s1_norm   = s1_grad.unitScale(
            s1_grad.reduceRegion(ee.Reducer.percentile([5]),  AOI, 100).values().get(0),
            s1_grad.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('score')
        S1_DISPONIBLE = True
        Fc_s1 = max(0, 1 - s1_antiguedad_dias / FC_S1_MAX_DIAS)
        print(f"  S1 OK -- antiguedad: {s1_antiguedad_dias}d | Fc: {Fc_s1:.2f}")
    else:
        print("  S1 sin dato reciente")
except Exception as e:
    print(f"  S1 error: {e}")

# ALOS-2
try:
    fecha_alos_ini = (FECHA_HOY - timedelta(days=FC_ALOS2_MAX_DIAS + 5)).strftime("%Y-%m-%d")
    alos2_col = (ee.ImageCollection(GEE_ALOS2)
                   .filterBounds(AOI)
                   .filterDate(fecha_alos_ini, FECHA_HOY_STR)
                   .select(['HH']))

    if alos2_col.size().getInfo() > 0:
        alos2_img  = alos2_col.sort('system:time_start', False).first()
        alos2_ts   = datetime.fromtimestamp(
            alos2_img.get('system:time_start').getInfo() / 1000)
        alos2_ant  = (FECHA_HOY - alos2_ts).days
        alos2_smooth = alos2_img.focal_mean(radius=3, kernelType='circle', units='pixels')
        alos2_grad   = alos2_smooth.gradient().abs().reduce(ee.Reducer.max())
        alos2_norm   = alos2_grad.unitScale(
            alos2_grad.reduceRegion(ee.Reducer.percentile([5]),  AOI, 100).values().get(0),
            alos2_grad.reduceRegion(ee.Reducer.percentile([95]), AOI, 100).values().get(0)
        ).rename('score')
        ALOS2_DISPONIBLE = True
        Fc_alos2 = max(0, 1 - alos2_ant / FC_ALOS2_MAX_DIAS)
        print(f"  ALOS-2 OK -- antiguedad: {alos2_ant}d | Fc: {Fc_alos2:.2f}")
    else:
        print("  ALOS-2 sin dato reciente")
except Exception as e:
    print(f"  ALOS-2 error: {e}")

# Batimetria
try:
    gebco      = ee.Image(GEE_GEBCO).select(GEE_GEBCO_BAND).clip(AOI)
    gebco_slope = ee.Terrain.slope(gebco).rename('score')
    print("  Batimetria OK")
except Exception as e:
    print(f"  Batimetria error: {e}")

sys.stdout.flush()

# ================================================================
# ESLABON 4 -- FUSION CADENA TROFICA
# Combina los 3 eslabones con pesos adaptativos
# ================================================================
print("\n[ESLABON 4/4] Fusion cadena trofica...")
sys.stdout.flush()

# Contar cuantas fuentes biologicas activas tenemos
fuentes_bio = sum([SST_DISPONIBLE, S2_BLOOM_DISPONIBLE, ERA5_DISPONIBLE])
fuentes_sar = sum([S1_DISPONIBLE, ALOS2_DISPONIBLE])
fuentes_total = fuentes_bio + fuentes_sar + (1 if gebco_slope else 0)

print(f"  Fuentes biologicas: {fuentes_bio} | SAR: {fuentes_sar} | Total: {fuentes_total}")

# Pesos adaptativos -- horizontal, complementario
# Si una fuente no esta, su peso se redistribuye entre las demas
W_base = {
    "sst_grad":  0.25,  # surgencia T-7d
    "chl_bloom": 0.20,  # bloom T-4d (S2)
    "viento":    0.15,  # surgencia proxy ERA5
    "s1_sar":    0.20,  # rugosidad actual
    "alos2":     0.10,  # estructura fondo
    "batimetria":0.10,  # pendiente fondo
}

# Calcular pesos activos
W_activos = {}
if SST_DISPONIBLE:      W_activos["sst_grad"]  = W_base["sst_grad"]
if S2_BLOOM_DISPONIBLE: W_activos["chl_bloom"] = W_base["chl_bloom"] * Fc_s2_bloom
if ERA5_DISPONIBLE:     W_activos["viento"]    = W_base["viento"]
if S1_DISPONIBLE:       W_activos["s1_sar"]    = W_base["s1_sar"] * Fc_s1
if ALOS2_DISPONIBLE:    W_activos["alos2"]     = W_base["alos2"] * Fc_alos2
if gebco_slope:         W_activos["batimetria"]= W_base["batimetria"]

# Normalizar para que sumen 1.0
total_w = sum(W_activos.values())
if total_w > 0:
    W_activos = {k: v/total_w for k, v in W_activos.items()}

print(f"  Pesos normalizados: {', '.join([f'{k}:{v:.2f}' for k,v in W_activos.items()])}")

# Determinar confianza
if fuentes_bio >= 2 and fuentes_sar >= 1:
    CONFIANZA = "ALTA"
elif fuentes_bio >= 1 or fuentes_sar >= 1:
    CONFIANZA = "MEDIA"
else:
    CONFIANZA = "BAJA"

print(f"  Confianza: {CONFIANZA}")
sys.stdout.flush()

# ================================================================
# CALCULO DEL MICROSCORE POR ZONA
# ================================================================
print("\nCalculando MicroScore por zona...")
sys.stdout.flush()

lats = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0045)
lons = np.arange(LON_MIN_C, LON_MAX_C, 0.0045)
grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')
puntos_flat = [(float(la), float(lo))
               for la, lo in zip(grid_lats.ravel(), grid_lons.ravel())]

def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(dlat**2 + dlon**2)

print(f"  Evaluando {len(puntos_flat)} puntos...")
sys.stdout.flush()

resultados = []

for i in range(0, min(len(puntos_flat), 500), 100):
    batch = puntos_flat[i:i+100]
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
        fc    = ee.FeatureCollection(features)
        capas = []

        # Capa SAR Sentinel-1
        if S1_DISPONIBLE and s1_norm:
            capas.append(s1_norm.multiply(W_activos.get("s1_sar", 0)))

        # Capa ALOS-2
        if ALOS2_DISPONIBLE and alos2_norm:
            capas.append(alos2_norm.multiply(W_activos.get("alos2", 0)))

        # Capa batimetria
        if gebco_slope:
            capas.append(gebco_slope.unitScale(0, 30).multiply(W_activos.get("batimetria", 0)))

        # Capa S2 bloom
        if S2_BLOOM_DISPONIBLE and chl_bloom_imagen:
            capas.append(chl_bloom_imagen.unitScale(0.8, 1.5).multiply(W_activos.get("chl_bloom", 0)))

        if not capas:
            continue

        score_img = ee.ImageCollection([c.rename('score') for c in capas]).sum().rename('score')

        # Factor ERA5 viento -- multiplicador global
        factor_viento = 1.0
        if ERA5_DISPONIBLE and viento_surgencia:
            factor_viento = 0.7 + 0.6 * viento_surgencia

        # Factor SST gradiente -- extraido como escalar por zona
        factor_sst = 1.0
        if SST_DISPONIBLE and sst_grad_norm is not None:
            factor_sst = 0.8 + 0.4 * float(np.nanmean(sst_grad_norm))

        res = score_img.reduceRegions(
            collection = fc,
            reducer    = ee.Reducer.mean(),
            scale      = 30
        ).getInfo()

        for feat in res['features']:
            p = feat['properties']
            if p.get('mean') is not None:
                score_base = float(p['mean'])
                score_final = score_base * factor_viento * factor_sst
                score_final = float(np.clip(score_final, 0, 1))
                resultados.append({
                    'lat':            p['lat'],
                    'lon':            p['lon'],
                    'dist_km':        round(p['dist_km'], 1),
                    'score':          round(score_final, 4),
                    'score_base':     round(score_base, 4),
                    'factor_viento':  round(factor_viento, 3),
                    'factor_sst':     round(factor_sst, 3),
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
    print("Sin resultados -- pipeline termina sin exportar")
    sys.exit(0)

df = pd.DataFrame(resultados).sort_values('score', ascending=False).reset_index(drop=True)

def get_semaforo(score):
    if score >= UMBRAL_VERDE:      return "VERDE"
    elif score >= UMBRAL_AMARILLO: return "AMARILLO"
    else:                          return "ROJO"

anillos = [(0,5),(5,10),(10,15),(15,20)]
df_reporte = []
for d_min, d_max in anillos:
    anillo = df[(df['dist_km'] >= d_min) & (df['dist_km'] < d_max)].head(6)
    df_reporte.append(anillo)
    print(f"  Anillo {d_min}-{d_max}km: {len(anillo)} zonas")

df_reporte = pd.concat(df_reporte, ignore_index=True)
df_reporte['semaforo']          = df_reporte['score'].apply(get_semaforo)
df_reporte['confianza']         = CONFIANZA
df_reporte['fecha']             = FECHA_HOY_STR
df_reporte['hora_utc']          = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
df_reporte['s1_dias']           = s1_antiguedad_dias if S1_DISPONIBLE else -1
df_reporte['sst_ok']            = SST_DISPONIBLE
df_reporte['s2_bloom_ok']       = S2_BLOOM_DISPONIBLE
df_reporte['era5_ok']           = ERA5_DISPONIBLE
df_reporte['alos2_ok']          = ALOS2_DISPONIBLE
df_reporte['sst_temp_medio']    = round(sst_valor_medio, 2) if sst_valor_medio else None
df_reporte['indice_surgencia']  = round(viento_surgencia, 3) if viento_surgencia else None

try:
    ws = sh.worksheet('costero_reporte')
    ws.clear()
except:
    ws = sh.add_worksheet(title='costero_reporte', rows=200, cols=25)
set_with_dataframe(ws, df_reporte)
print(f"Reporte exportado: {len(df_reporte)} zonas")
sys.stdout.flush()

# ================================================================
# HISTORIAL 90 DIAS
# ================================================================
print("\nActualizando historial...")
sys.stdout.flush()

try:
    ws_hist = sh.worksheet('costero_historial')
    df_hist = pd.DataFrame(ws_hist.get_all_records())
except:
    ws_hist = sh.add_worksheet(title='costero_historial', rows=5000, cols=25)
    df_hist = pd.DataFrame()

cols_hist = ['lat','lon','dist_km','score','semaforo','confianza',
             'fecha','hora_utc','s1_dias','sst_ok','s2_bloom_ok',
             'era5_ok','indice_surgencia']
df_hist_nuevo = df_reporte[[c for c in cols_hist if c in df_reporte.columns]].copy()

if len(df_hist) > 0:
    df_hist_total = pd.concat([df_hist, df_hist_nuevo], ignore_index=True)
else:
    df_hist_total = df_hist_nuevo

fecha_limite  = (datetime.utcnow() - timedelta(days=HISTORIAL_DIAS)).strftime("%Y-%m-%d")
df_hist_total = df_hist_total[df_hist_total['fecha'] >= fecha_limite].reset_index(drop=True)
ws_hist.clear()
set_with_dataframe(ws_hist, df_hist_total)
print(f"Historial: {len(df_hist_total)} registros")
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
        'score': float(zona['score']),
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
    ws_ipo = sh.add_worksheet(title='costero_ipo', rows=200, cols=15)
set_with_dataframe(ws_ipo, df_ipo)
print(f"IPO: {len(df_ipo)} zonas")
sys.stdout.flush()

# ================================================================
# RESUMEN FINAL
# ================================================================
print("\n" + "=" * 60)
print(f"PredictaMAR Costero v1.1 COMPLETO -- {FECHA_HOY_STR}")
print(f"Confianza: {CONFIANZA}")
print(f"Eslabon 1 - Surgencia SST:  {'OK' if SST_DISPONIBLE else 'NO'}")
print(f"Eslabon 2 - Bloom S2:       {'OK Fc=' + str(round(Fc_s2_bloom,2)) if S2_BLOOM_DISPONIBLE else 'NO'}")
print(f"Eslabon 2 - Viento ERA5:    {'OK ind=' + str(round(viento_surgencia,2)) if ERA5_DISPONIBLE else 'NO'}")
print(f"Eslabon 3 - SAR S1:         {'OK Fc=' + str(round(Fc_s1,2)) if S1_DISPONIBLE else 'NO'}")
print(f"Eslabon 3 - ALOS-2:         {'OK Fc=' + str(round(Fc_alos2,2)) if ALOS2_DISPONIBLE else 'NO'}")
print(f"Eslabon 3 - Batimetria:     {'OK' if gebco_slope else 'NO'}")
print(f"Zonas reportadas: {len(df_reporte)}")
print(f"Fin: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()
