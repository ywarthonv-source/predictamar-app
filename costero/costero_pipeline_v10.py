# ================================================================
# PredictaMAR Costero v1.1 -- PIPELINE COMPLETO
# Puerto Chorrillos - 0-20 km - Sistema Corriente de Humboldt
#
# Cadena trofica temporal:
#   surgencia(T-7d, SST+viento) -> bloom(T-4d, S2) -> cardumen(hoy)
#
# Variables score base (pesos adaptativos horizontales):
#   SST gradiente 18% | ERA5 viento 12% | S2 bloom 15%
#   S1 SAR 15% | ALOS-2 10% | Batimetria 15% | Geometria costera 15%
#
# Moduladores externos (fuera del score base):
#   Oleaje: kill-switch SWH>1.5m + penalizacion mezcla
#   Corrientes: adveccion posicion proyectada T8h y T16h
# ================================================================

import sys, os, json
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials as SACredentials
from datetime import datetime, timedelta
import copernicusmarine
import ee
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PredictaMAR Costero v1.1 -- PIPELINE COMPLETO")
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
print(f"DEBUG SHEET_ID: {SHEET_ID[:8]}...")
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
sys.stdout.flush()

AOI = ee.Geometry.Rectangle([LON_MIN_C, LAT_MIN_C, LON_MAX_C, LAT_MAX_C])

def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(dlat**2 + dlon**2)

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
print("\n[3/8] Oleaje CMEMS/GEE -- condicion del mar...")
OLEAJE_DISPONIBLE  = False
swh_medio          = 0.8
kill_switch_activo = False
penalizacion_mezcla = 1.0

try:
    fecha_ola_ini = (FECHA_HOY - timedelta(days=1)).strftime("%Y-%m-%d")
    waves_col = (ee.ImageCollection(GEE_WAVES)
                   .filterBounds(AOI)
                   .filterDate(fecha_ola_ini, FECHA_HOY_STR)
                   .select(['VHM0']))

    if waves_col.size().getInfo() > 0:
        waves_img = waves_col.sort('system:time_start', False).first()
        swh_stats = waves_img.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = AOI,
            scale     = 9000
        ).getInfo()
        swh_val = swh_stats.get('VHM0')
        if swh_val is not None:
            swh_medio = float(swh_val)
            OLEAJE_DISPONIBLE = True

            # Kill-switch operacional
            if swh_medio > SWH_KILL_SWITCH:
                kill_switch_activo = True
                print(f"  OLEAJE ADVERSO -- SWH: {swh_medio:.2f}m > {SWH_KILL_SWITCH}m -- KILL SWITCH ACTIVO")
            else:
                # Penalizacion mezcla: escala entre 0 y SWH_MEZCLA_MAX
                penalizacion_mezcla = max(0.3,
                    1.0 - (swh_medio / SWH_MEZCLA_MAX) * 0.7)
                print(f"  Oleaje OK -- SWH: {swh_medio:.2f}m | pen_mezcla: {penalizacion_mezcla:.2f}")
        else:
            print("  Oleaje sin dato en AOI")
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
# FUENTE 5 -- SENTINEL-2 (bloom T-4d)
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
        Fc_s2 = max(0, 1 - dias_diff / 5)
        print(f"  S2 OK -- fecha: {s2_ts.strftime('%Y-%m-%d')} | Fc: {Fc_s2:.2f}")
    else:
        print("  S2 sin dato -- nubosidad")
except Exception as e:
    print(f"  S2 error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 6 -- SENTINEL-1 SAR (condiciones actuales)
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
        Fc_s1 = max(0, 1 - s1_ant / FC_S1_MAX_DIAS)
        print(f"  S1 OK -- antiguedad: {s1_ant}d | Fc: {Fc_s1:.2f}")
    else:
        print("  S1 sin dato reciente")
except Exception as e:
    print(f"  S1 error: {e}")
sys.stdout.flush()

# ================================================================
# FUENTE 7 -- ALOS-2 (estructura fondo banda L)
# ================================================================
print("\n[7/8] ALOS-2 + batimetria + geometria costera...")
ALOS2_DISPONIBLE = False
alos2_norm       = None
Fc_alos2         = 0.0
gebco_slope      = None
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
        Fc_alos2 = max(0, 1 - alos2_ant / FC_ALOS2_MAX_DIAS)
        print(f"  ALOS-2 OK -- antiguedad: {alos2_ant}d | Fc: {Fc_alos2:.2f}")
    else:
        print("  ALOS-2 sin dato reciente")
except Exception as e:
    print(f"  ALOS-2 error: {e}")

# ================================================================
# FUENTE 8 -- BATIMETRIA + GEOMETRIA COSTERA
# Batimetria: pendiente del fondo (ETOPO1)
# Geometria costera: gradiente batimetrico perpendicular a la costa
# Detecta puntas, canyones y zonas de retencion mecanica
# (Morro Solar, La Punta) donde la corriente se desacelera
# y atrapa zooplancton y nutrientes
# ================================================================
try:
    gebco = ee.Image(GEE_GEBCO).select(GEE_GEBCO_BAND).clip(AOI)

    # Pendiente del fondo -- proxy habitat demersal
    gebco_slope = ee.Terrain.slope(gebco).rename('score')

    # Gradiente batimetrico perpendicular a la costa
    # Calcula la segunda derivada del fondo -- donde hay cambios
    # abruptos de pendiente = puntas, canyones, zonas de retencion
    gebco_smooth = gebco.focal_mean(radius=3, kernelType='circle', units='pixels')
    grad_x       = gebco_smooth.gradient().select('x')
    grad_y       = gebco_smooth.gradient().select('y')
    grad_mag     = grad_x.pow(2).add(grad_y.pow(2)).sqrt()
    grad2_x      = grad_mag.gradient().select('x')
    grad2_y      = grad_mag.gradient().select('y')
    curv_bati    = grad2_x.pow(2).add(grad2_y.pow(2)).sqrt()

    # Normalizar -- valores altos = zonas de retencion geometrica
    geometria_costera = curv_bati.unitScale(
        curv_bati.reduceRegion(ee.Reducer.percentile([5]),  AOI, 500).values().get(0),
        curv_bati.reduceRegion(ee.Reducer.percentile([95]), AOI, 500).values().get(0)
    ).rename('score')

    print("  Batimetria + geometria costera OK")
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
    W_activos = {k: v/total_w for k, v in W_activos.items()}

n_bio = sum([SST_DISPONIBLE, ERA5_DISPONIBLE, S2_DISPONIBLE])
n_fis = sum([S1_DISPONIBLE, ALOS2_DISPONIBLE, bool(gebco_slope), bool(geometria_costera)])
CONFIANZA = "ALTA" if n_bio >= 2 and n_fis >= 2 else "MEDIA" if n_bio >= 1 or n_fis >= 2 else "BAJA"

print(f"  Bio: {n_bio} | Fisico: {n_fis} | Confianza: {CONFIANZA}")
print(f"  Pesos: {', '.join([f'{k[:6]}:{v:.2f}' for k,v in W_activos.items()])}")
if kill_switch_activo:
    print(f"  KILL SWITCH ACTIVO -- SWH {swh_medio:.2f}m > {SWH_KILL_SWITCH}m")
sys.stdout.flush()

# ================================================================
# CALCULO DEL MICROSCORE
# ================================================================
print("\nCalculando MicroScore...")

lats = np.arange(LAT_MIN_C, LAT_MAX_C, 0.0045)
lons = np.arange(LON_MIN_C, LON_MAX_C, 0.0045)
grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')
puntos_flat = [(float(la), float(lo))
               for la, lo in zip(grid_lats.ravel(), grid_lons.ravel())]

def adveccion_punto(lat, lon, horas):
    if not CORRIENTES_DISPONIBLE:
        return lat, lon
    segundos = horas * 3600
    dlat = (vo_medio * segundos) / 111000
    dlon = (uo_medio * segundos) / (111000 * np.cos(np.radians(lat)))
    return round(lat + dlat, 4), round(lon + dlon, 4)

print(f"  Evaluando {len(puntos_flat)} puntos...")
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

        if S1_DISPONIBLE and s1_norm:
            capas.append(s1_norm.multiply(W_activos.get("s1_sar", 0)))
        if ALOS2_DISPONIBLE and alos2_norm:
            capas.append(alos2_norm.multiply(W_activos.get("alos2", 0)))
        if gebco_slope:
            capas.append(gebco_slope.unitScale(0, 30).multiply(W_activos.get("batimetria", 0)))
        if geometria_costera:
            capas.append(geometria_costera.multiply(W_activos.get("geometria", 0)))
        if S2_DISPONIBLE and chl_bloom_imagen:
            capas.append(chl_bloom_imagen.unitScale(0.8, 1.5).multiply(W_activos.get("chl_bloom", 0)))
        if not capas:
            continue

        score_img = ee.ImageCollection([c.rename('score') for c in capas]).sum().rename('score')
        res = score_img.reduceRegions(
            collection = fc,
            reducer    = ee.Reducer.mean(),
            scale      = 30
        ).getInfo()

        for feat in res['features']:
            p = feat['properties']
            if p.get('mean') is not None:
                lat_p      = p['lat']
                lon_p      = p['lon']
                score_base = float(p['mean'])

                # Multiplicadores biologicos escalares
                factor_sst    = 0.8 + 0.4 * sst_grad_medio
                factor_viento = 0.7 + 0.6 * indice_surgencia if ERA5_DISPONIBLE else 1.0

                # Modulador oleaje (penalizacion mezcla vertical)
                # Kill-switch ya se aplica al semaforo, no al score
                factor_oleaje = penalizacion_mezcla

                score_final = float(np.clip(
                    score_base * factor_sst * factor_viento * factor_oleaje,
                    0, 1
                ))

                # Adveccion: posicion proyectada
                lat_8h,  lon_8h  = adveccion_punto(lat_p, lon_p, ADV_HORAS_T8)
                lat_16h, lon_16h = adveccion_punto(lat_p, lon_p, ADV_HORAS_T16)
                dist_16h = dist_km(LAT_CHORRILLOS, LON_CHORRILLOS, lat_16h, lon_16h)
                desp_km  = dist_km(lat_p, lon_p, lat_16h, lon_16h)
                angulo   = float(np.degrees(np.arctan2(lon_16h - lon_p, lat_16h - lat_p)))
                dirs     = ["N","NE","E","SE","S","SO","O","NO"]
                dir_txt  = dirs[int((angulo + 22.5) / 45) % 8]

                resultados.append({
                    'lat':         lat_p,
                    'lon':         lon_p,
                    'dist_km':     round(p['dist_km'], 1),
                    'score':       round(score_final, 4),
                    'score_base':  round(score_base, 4),
                    'lat_T8':      lat_8h,
                    'lon_T8':      lon_8h,
                    'lat_T16':     lat_16h,
                    'lon_T16':     lon_16h,
                    'dist_T16':    round(dist_16h, 1),
                    'desp_km':     round(desp_km, 2),
                    'direccion':   dir_txt,
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

if not resultados:
    print("Sin resultados -- pipeline termina sin exportar")
    sys.exit(0)

df = pd.DataFrame(resultados).sort_values('score', ascending=False).reset_index(drop=True)

def get_semaforo(score, kill_switch):
    if kill_switch:            return "ADVERSO"
    if score >= UMBRAL_VERDE:  return "VERDE"
    if score >= UMBRAL_AMARILLO: return "AMARILLO"
    return "ROJO"

anillos = [(0,5),(5,10),(10,15),(15,20)]
df_rep  = []
for d_min, d_max in anillos:
    anillo = df[(df['dist_km'] >= d_min) & (df['dist_km'] < d_max)].head(6)
    df_rep.append(anillo)
    print(f"  Anillo {d_min}-{d_max}km: {len(anillo)} zonas")

df_rep = pd.concat(df_rep, ignore_index=True)
df_rep['semaforo']          = df_rep['score'].apply(
    lambda s: get_semaforo(s, kill_switch_activo))
df_rep['confianza']         = CONFIANZA
df_rep['fecha']             = FECHA_HOY_STR
df_rep['hora_utc']          = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
df_rep['s1_dias']           = s1_ant if S1_DISPONIBLE else -1
df_rep['sst_ok']            = SST_DISPONIBLE
df_rep['s2_bloom_ok']       = S2_DISPONIBLE
df_rep['era5_ok']           = ERA5_DISPONIBLE
df_rep['corrientes_ok']     = CORRIENTES_DISPONIBLE
df_rep['alos2_ok']          = ALOS2_DISPONIBLE
df_rep['oleaje_ok']         = OLEAJE_DISPONIBLE
df_rep['swh_medio']         = round(swh_medio, 2)
df_rep['kill_switch']       = kill_switch_activo
df_rep['sst_temp_medio']    = round(sst_temp_medio, 2) if sst_temp_medio else None
df_rep['indice_surgencia']  = round(indice_surgencia, 3)
df_rep['uo_medio']          = round(uo_medio, 5)
df_rep['vo_medio']          = round(vo_medio, 5)

try:
    ws = sh.worksheet('costero_reporte')
    ws.clear()
except:
    ws = sh.add_worksheet(title='costero_reporte', rows=200, cols=35)
set_with_dataframe(ws, df_rep)
print(f"Reporte exportado: {len(df_rep)} zonas")
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

cols_h = ['lat','lon','dist_km','score','semaforo','confianza',
          'fecha','hora_utc','s1_dias','sst_ok','s2_bloom_ok',
          'era5_ok','swh_medio','kill_switch','indice_surgencia','direccion']
df_h_nuevo    = df_rep[[c for c in cols_h if c in df_rep.columns]].copy()
df_hist_total = pd.concat([df_hist, df_h_nuevo], ignore_index=True) if len(df_hist) > 0 else df_h_nuevo
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
        'score': float(zona['score']),
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
print(f"SST L4:      {'OK T=' + str(round(sst_temp_medio,1)) if SST_DISPONIBLE else 'NO'}")
print(f"ERA5:        {'OK ind=' + str(round(indice_surgencia,2)) if ERA5_DISPONIBLE else 'NO'}")
print(f"S2 bloom:    {'OK Fc=' + str(round(Fc_s2,2)) if S2_DISPONIBLE else 'NO (nubosidad)'}")
print(f"S1 SAR:      {'OK Fc=' + str(round(Fc_s1,2)) if S1_DISPONIBLE else 'NO'}")
print(f"ALOS-2:      {'OK Fc=' + str(round(Fc_alos2,2)) if ALOS2_DISPONIBLE else 'NO'}")
print(f"Corrientes:  {'OK' if CORRIENTES_DISPONIBLE else 'NO'}")
print(f"Oleaje:      {'ADVERSO SWH=' + str(round(swh_medio,2)) + 'm' if kill_switch_activo else 'OK SWH=' + str(round(swh_medio,2)) + 'm'}")
print(f"Batimetria:  {'OK' if gebco_slope else 'NO'}")
print(f"Geometria:   {'OK' if geometria_costera else 'NO'}")
print(f"Zonas: {len(df_rep)}")
print(f"Fin: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()
