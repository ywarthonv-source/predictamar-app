# ================================================================
# PredictaMAR v6.2 — PIPELINE AUTOMATICO
# Corre desde GitHub Actions cada 6 horas
# Lee credenciales desde variables de entorno
# ================================================================

import sys
print(f"Python: {sys.version}")
print("Iniciando imports...")
sys.stdout.flush()

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

print("Imports OK")
sys.stdout.flush()

print("=" * 60)
print("PREDICTAMAR v6.2 — PIPELINE AUTOMATICO")
print(f"Inicio: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 60)
sys.stdout.flush()

# ── Credenciales desde variables de entorno ───────────────────────
CMEMS_USER   = os.environ["CMEMS_USER"]
CMEMS_PASS   = os.environ["CMEMS_PASS"]
NASA_TOKEN   = os.environ["NASA_TOKEN"]
CDS_KEY      = os.environ["CDS_KEY"]
SHEET_ID     = os.environ["SHEET_ID"]
GOOGLE_SA    = os.environ["GOOGLE_SA_JSON"]

print("Credenciales cargadas OK")
sys.stdout.flush()

headers_nasa = {"Authorization": f"Bearer {NASA_TOKEN}"}

# CDS
with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
    f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {CDS_KEY}\n")

# Google Sheets via Service Account
sa_info  = json.loads(GOOGLE_SA)
scopes   = ["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]
sa_creds = SACredentials.from_service_account_info(sa_info, scopes=scopes)
gc       = gspread.authorize(sa_creds)
sh       = gc.open_by_key(SHEET_ID)
print(f"Sheets OK: {sh.title}")
sys.stdout.flush()

# CMEMS login
copernicusmarine.login(
    username=CMEMS_USER,
    password=CMEMS_PASS,
    force_overwrite=True
)
print("CMEMS OK")
sys.stdout.flush()

# GEE via Service Account
gee_creds = ee.ServiceAccountCredentials(
    sa_info["client_email"],
    key_data=sa_info["private_key"]
)
ee.Initialize(gee_creds)
print("GEE OK")
sys.stdout.flush()

# ── Parametros ────────────────────────────────────────────────────
LAT_MIN, LAT_MAX   = -22.0, -3.0
LON_MIN, LON_MAX   = -85.0, -68.0
LAT_CHRISTIAN      = -12.15
LON_CHRISTIAN      = -77.02
UMBRAL_VERDE       = 0.62
UMBRAL_AMARILLO    = 0.52
UMBRAL_S2          = 0.60
BUFFER_M           = 3000
W_S3, W_S2         = 0.80, 0.20
DRIVE_BASE         = "/tmp/predictamar"
os.makedirs(f"{DRIVE_BASE}/raw", exist_ok=True)

FECHA_FIN     = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
FECHA_FIN_STR = FECHA_FIN.strftime("%Y-%m-%d")
print(f"Fecha: {FECHA_FIN_STR}")
sys.stdout.flush()

W = {"chl": 0.30, "sst": 0.20, "front": 0.15,
     "ic": 0.15, "sla": 0.12, "grad_sla": 0.08}

# ── Funciones utiles ──────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1+lat2)/2))
    return np.sqrt(dlat**2 + dlon**2)

def penalizacion_distancia(dist_km):
    pen = np.ones_like(dist_km, dtype=float)
    pen[(dist_km > 30)  & (dist_km <= 60)]  = 0.80
    pen[(dist_km > 60)  & (dist_km <= 100)] = 0.55
    pen[dist_km > 100]  = 0.30
    return pen

def gradiente_magnitud(campo, res_km=4.0):
    campo_clean = np.where(np.isnan(campo), 0, campo)
    gy, gx = np.gradient(campo_clean, res_km, res_km)
    grad = np.sqrt(gx**2 + gy**2)
    grad[np.isnan(campo)] = np.nan
    return grad

def normalizar_percentil(arr, p_low=10, p_high=90):
    validos = arr[~np.isnan(arr)]
    v_low   = np.nanpercentile(validos, p_low)
    v_high  = np.nanpercentile(validos, p_high)
    norm    = np.clip((arr - v_low) / max(v_high - v_low, 1e-6), 0, 1)
    norm[np.isnan(arr)] = np.nan
    return norm

def calcular_desplazamiento(uo, vo, horas, lat_ref=-12.0):
    segundos = horas * 3600
    dlat_deg = (vo * segundos) / 111000
    dlon_deg = (uo * segundos) / (111000 * np.cos(np.radians(lat_ref)))
    return dlat_deg, dlon_deg

def get_semaforo(score):
    if score >= UMBRAL_VERDE:      return "VERDE"
    elif score >= UMBRAL_AMARILLO: return "AMARILLO"
    else:                          return "ROJO"

# ── CMEMS descarga ────────────────────────────────────────────────
def descargar_cmems(dataset_id, variables, nombre, depth=True):
    out = f"{DRIVE_BASE}/raw/{nombre}.nc"
    if os.path.exists(out): os.remove(out)
    kwargs = dict(
        dataset_id        = dataset_id,
        variables         = variables,
        minimum_latitude  = LAT_MIN,
        maximum_latitude  = LAT_MAX,
        minimum_longitude = LON_MIN,
        maximum_longitude = LON_MAX,
        start_datetime    = (FECHA_FIN - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime      = FECHA_FIN.strftime("%Y-%m-%dT23:59:59"),
        output_filename   = f"{nombre}.nc",
        output_directory  = f"{DRIVE_BASE}/raw",
        username          = CMEMS_USER,
        password          = CMEMS_PASS,
        disable_progress_bar = True
    )
    if depth:
        kwargs["minimum_depth"] = 0.49
        kwargs["maximum_depth"] = 0.51
    copernicusmarine.subset(**kwargs)
    ds = xr.open_dataset(out)
    print(f"  OK — {nombre}: {dict(ds.dims)}")
    sys.stdout.flush()
    return ds

print("\nDescargando CMEMS...")
sys.stdout.flush()
ds_olci = descargar_cmems("cmems_obs-oc_glo_bgc-plankton_nrt_l3-olci-300m_P1D", ["CHL"], "v6_olci", depth=False)
ds_sst  = descargar_cmems("cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i", ["thetao"], "v6_sst")
ds_sla  = descargar_cmems("cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D", ["sla"], "v6_sla", depth=False)
ds_cur  = descargar_cmems("cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i", ["uo", "vo"], "v6_cur")
print("CMEMS completo.")
sys.stdout.flush()

# ── VIIRS NASA ────────────────────────────────────────────────────
print("\nDescargando VIIRS...")
sys.stdout.flush()
VIIRS_DISPONIBLE = False
viirs_datasets   = []

url_viirs = (
    f"https://cmr.earthdata.nasa.gov/search/granules.json"
    f"?short_name=VIIRSJ1_L3m_CHL_NRT"
    f"&bounding_box={LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"
    f"&temporal={(FECHA_FIN - timedelta(days=7)).strftime('%Y-%m-%d')}T00:00:00Z,"
    f"{FECHA_FIN.strftime('%Y-%m-%d')}T23:59:59Z"
    f"&page_size=10&sort_key=-start_date"
)
resp_v         = requests.get(url_viirs, headers=headers_nasa, timeout=30)
granulas_viirs = resp_v.json().get("feed", {}).get("entry", [])
granulas_4km   = [g for g in granulas_viirs if "4km" in g.get("title","")]

for g in granulas_4km[:3]:
    fecha = g.get("time_start","")[:10]
    links = g.get("links", [])
    link  = next((l["href"] for l in links
                  if "https" in l.get("href","") and ".nc" in l.get("href","")), None)
    if not link: continue
    viirs_path = f"{DRIVE_BASE}/raw/viirs_{fecha}.nc"
    r = requests.get(link, headers=headers_nasa, stream=True, timeout=180)
    if r.status_code == 200:
        with open(viirs_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        try:
            ds_tmp = xr.open_dataset(viirs_path)
            var    = "chlor_a" if "chlor_a" in ds_tmp else list(ds_tmp.data_vars)[0]
            viirs_datasets.append({'ds': ds_tmp, 'fecha': fecha, 'var': var})
            VIIRS_DISPONIBLE = True
            print(f"  VIIRS OK: {fecha}")
            sys.stdout.flush()
        except: pass

print(f"VIIRS: {'OK' if VIIRS_DISPONIBLE else 'Fallback CMEMS'}")
sys.stdout.flush()

# ── Grilla OLCI ───────────────────────────────────────────────────
print("\nConstruyendo grilla...")
sys.stdout.flush()
LAT_REF = ds_olci['latitude'].values
LON_REF = ds_olci['longitude'].values

chl_raw         = ds_olci['CHL'].values
chl_mean        = np.nanmean(chl_raw, axis=0)
chl_std         = np.nanstd(chl_raw, axis=0)
chl_mean_masked = np.where(chl_mean >= 0.1, chl_mean, np.nan)
chl_std_masked  = np.where(chl_mean >= 0.1, chl_std, np.nan)

Ic_v2    = chl_mean_masked / (chl_mean_masked + chl_std_masked + 1e-9)
Ic_v2    = np.clip(Ic_v2, 0, 1)
Ic_v2    = np.where(np.isnan(chl_mean_masked), np.nan, Ic_v2)

CHL_P95  = np.nanpercentile(chl_mean_masked[~np.isnan(chl_mean_masked)], 95)
CHL_P05  = np.nanpercentile(chl_mean_masked[~np.isnan(chl_mean_masked)], 5)
chl_norm = np.clip((chl_mean_masked - CHL_P05) / (CHL_P95 - CHL_P05 + 1e-9), 0, 1)

def interpolar_a_olci(ds, var):
    try:
        da = ds[var]
        if "depth" in da.dims: da = da.isel(depth=0)
        da_mean  = da.mean(dim="time")
        lat_name = "latitude" if "latitude" in da_mean.dims else "lat"
        lon_name = "longitude" if "longitude" in da_mean.dims else "lon"
        return da_mean.interp(
            {lat_name: xr.DataArray(LAT_REF, dims="lat"),
             lon_name: xr.DataArray(LON_REF, dims="lon")},
            method="linear"
        ).values
    except Exception as e:
        print(f"  Error {var}: {e}")
        return np.full((len(LAT_REF), len(LON_REF)), np.nan)

sst_grid = interpolar_a_olci(ds_sst, "thetao")
sla_grid = interpolar_a_olci(ds_sla, "sla")
uo_grid  = interpolar_a_olci(ds_cur, "uo")
vo_grid  = interpolar_a_olci(ds_cur, "vo")

uo_mean_val = float(np.nanmean(uo_grid))
vo_mean_val = float(np.nanmean(vo_grid))

SST_P05  = np.nanpercentile(sst_grid[~np.isnan(sst_grid)], 5)
SST_P95  = np.nanpercentile(sst_grid[~np.isnan(sst_grid)], 95)
sst_norm = np.clip((SST_P95 - sst_grid) / (SST_P95 - SST_P05 + 1e-9), 0, 1)
sst_norm[np.isnan(sst_grid)] = np.nan

sla_inv  = -sla_grid
SLA_P05  = np.nanpercentile(sla_inv[~np.isnan(sla_inv)], 5)
SLA_P95  = np.nanpercentile(sla_inv[~np.isnan(sla_inv)], 95)
sla_norm = np.clip((sla_inv - SLA_P05) / (SLA_P95 - SLA_P05 + 1e-9), 0, 1)
sla_norm[np.isnan(sla_grid)] = np.nan

grad_chl    = gradiente_magnitud(chl_mean_masked, 0.3)
grad_sst    = gradiente_magnitud(sst_grid, 4.0)
grad_sla    = gradiente_magnitud(sla_grid, 12.0)
sc_grad_chl = normalizar_percentil(grad_chl)
sc_grad_sst = normalizar_percentil(grad_sst)
sc_grad_sla = normalizar_percentil(grad_sla)
sc_front    = 0.6 * sc_grad_sst + 0.4 * sc_grad_chl
sc_front[np.isnan(chl_mean_masked)] = np.nan

macro_v62 = (
    W["chl"]      * np.nan_to_num(chl_norm,    nan=0) +
    W["sst"]      * np.nan_to_num(sst_norm,    nan=0) +
    W["front"]    * np.nan_to_num(sc_front,    nan=0) +
    W["ic"]       * np.nan_to_num(Ic_v2,       nan=0) +
    W["sla"]      * np.nan_to_num(sla_norm,    nan=0) +
    W["grad_sla"] * np.nan_to_num(sc_grad_sla, nan=0)
)

mask_valido   = (~np.isnan(chl_mean_masked) & ~np.isnan(sst_grid) & ~np.isnan(sla_grid))
macro_v62[~mask_valido] = np.nan
tierra_mask   = np.isnan(sst_grid)
dist_costa_km = distance_transform_edt(~tierra_mask) * 0.3
pen_dist      = penalizacion_distancia(dist_costa_km)
pen_dist[tierra_mask] = np.nan
macro_v62_pen = macro_v62 * pen_dist
macro_v62_pen[~mask_valido] = np.nan

print(f"Score max: {np.nanmax(macro_v62_pen):.3f}")
sys.stdout.flush()

# ── DataFrame ─────────────────────────────────────────────────────
lats, lons = np.meshgrid(LAT_REF, LON_REF, indexing='ij')
df_grilla  = pd.DataFrame({
    'LAT_REF':     lats.ravel(),
    'LON_REF':     lons.ravel(),
    'score_total': macro_v62_pen.ravel(),
    'sst':         sst_grid.ravel(),
    'sla':         sla_grid.ravel(),
    'chl':         chl_mean_masked.ravel(),
}).dropna(subset=['score_total']).reset_index(drop=True)

df_grilla = df_grilla[
    (df_grilla['LAT_REF'] >= LAT_MIN) & (df_grilla['LAT_REF'] <= LAT_MAX) &
    (df_grilla['LON_REF'] >= LON_MIN) & (df_grilla['LON_REF'] <= LON_MAX)
].reset_index(drop=True)

df_candidatos = df_grilla[df_grilla['score_total'] >= UMBRAL_S2].iloc[::10].reset_index(drop=True)
df_resto      = df_grilla[df_grilla['score_total'] <  UMBRAL_S2].copy()

print(f"Puntos validos: {len(df_grilla)} | Candidatos S2: {len(df_candidatos)}")
sys.stdout.flush()

# ── S2 GEE ────────────────────────────────────────────────────────
print(f"\nRefinamiento S2...")
sys.stdout.flush()

def mask_and_index(image):
    scl  = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    b4   = image.select('B4').toFloat().divide(10000)
    b8   = image.select('B8').toFloat().divide(10000)
    b11  = image.select('B11').toFloat().divide(10000)
    fai  = b8.subtract(b4).subtract(
               b11.subtract(b4).multiply((842-665)/(1610-665))
           ).rename('FAI')
    isb  = b8.divide(b11.add(1e-6)).rename('ISB')
    return image.addBands([fai, isb]).updateMask(mask)

s2_clean = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate((FECHA_FIN - timedelta(days=10)).strftime("%Y-%m-%d"), FECHA_FIN_STR)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .filterBounds(ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX]))
              .select(['B4','B8','B11','SCL'])
              .map(mask_and_index)
              .median())

resultados_s2 = []
for i in range(0, len(df_candidatos), 50):
    batch  = df_candidatos.iloc[i:i+50]
    puntos = []
    for _, r in batch.iterrows():
        try:
            puntos.append(ee.Feature(
                ee.Geometry.Point([float(r['LON_REF']), float(r['LAT_REF'])]).buffer(BUFFER_M),
                {'lat': float(r['LAT_REF']), 'lon': float(r['LON_REF']),
                 'score_s3': float(r['score_total'])}
            ))
        except: continue
    if not puntos: continue
    try:
        res = s2_clean.select(['FAI','ISB']).reduceRegions(
            collection=ee.FeatureCollection(puntos),
            reducer=ee.Reducer.mean(), scale=20
        ).getInfo()
        for feat in res['features']:
            p = feat['properties']
            resultados_s2.append({
                'LAT_REF':  p['lat'], 'LON_REF': p['lon'],
                'score_s3': p['score_s3'],
                'FAI':      p.get('FAI', np.nan),
                'ISB':      p.get('ISB', np.nan)
            })
    except: pass

df_s2 = pd.DataFrame(resultados_s2)
if len(df_s2) > 0:
    for col in ['FAI','ISB']:
        mn, mx = df_s2[col].min(), df_s2[col].max()
        df_s2[f'{col}_norm'] = (df_s2[col] - mn) / (mx - mn + 1e-9)
    df_s2['score_fusionado'] = (
        W_S3 * df_s2['score_s3'] +
        W_S2 * (df_s2['FAI_norm'] * 0.6 + df_s2['ISB_norm'] * 0.4)
    )
    df_s2_final = df_s2[['LAT_REF','LON_REF','score_fusionado']]
else:
    df_s2_final = pd.DataFrame(columns=['LAT_REF','LON_REF','score_fusionado'])

df_resto_final = df_resto[['LAT_REF','LON_REF','score_total']].rename(
    columns={'score_total': 'score_fusionado'}
)
df_final = pd.concat([df_s2_final, df_resto_final], ignore_index=True)
df_final = df_final.sort_values('score_fusionado', ascending=False).reset_index(drop=True)
df_final['fecha'] = FECHA_FIN_STR

print(f"Puntos finales: {len(df_final)}")
sys.stdout.flush()

# ── Adveccion ─────────────────────────────────────────────────────
lat_med  = df_final['LAT_REF'].mean()
dl8,  dn8  = calcular_desplazamiento(uo_mean_val, vo_mean_val, 8,  lat_ref=lat_med)
dl16, dn16 = calcular_desplazamiento(uo_mean_val, vo_mean_val, 16, lat_ref=lat_med)
dl24, dn24 = calcular_desplazamiento(uo_mean_val, vo_mean_val, 24, lat_ref=lat_med)

df_final['LAT_T8']   = df_final['LAT_REF'] + dl8
df_final['LON_T8']   = df_final['LON_REF'] + dn8
df_final['LAT_T16']  = df_final['LAT_REF'] + dl16
df_final['LON_T16']  = df_final['LON_REF'] + dn16
df_final['LAT_T24']  = df_final['LAT_REF'] + dl24
df_final['LON_T24']  = df_final['LON_REF'] + dn24
df_final['delta_km'] = np.sqrt((dl16*111)**2 + (dn16*111)**2)
df_final['dist_ch']  = df_final.apply(
    lambda r: distancia_km(LAT_CHRISTIAN, LON_CHRISTIAN, r['LAT_T16'], r['LON_T16']), axis=1
)

dlat_por_hora = dl16 / 16
dlon_por_hora = dn16 / 16

# ── Exportar reporte por anillos ──────────────────────────────────
print("\nExportando reporte por anillos...")
sys.stdout.flush()

df_validos = df_final[
    (df_final['dist_ch'] <= 80) &
    (df_final['score_fusionado'] >= 0.50)
].copy()

anillos = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80)]
df_distribuido = []

for d_min, d_max in anillos:
    anillo = df_validos[
        (df_validos['dist_ch'] >= d_min) &
        (df_validos['dist_ch'] <  d_max)
    ].nlargest(12, 'score_fusionado')
    df_distribuido.append(anillo)
    print(f"  Anillo {d_min}-{d_max}km: {len(anillo)} puntos")

df_reporte = pd.concat(df_distribuido, ignore_index=True)
df_reporte = df_reporte.sort_values('score_fusionado', ascending=False).reset_index(drop=True)
print(f"Total puntos: {len(df_reporte)}")
sys.stdout.flush()

reporte_rows = []
for i, row in df_reporte.iterrows():
    score   = float(row['score_fusionado'])
    sst_val = round(float(row['sst']), 2) if 'sst' in row and not pd.isna(row['sst']) else None
    chl_val = round(float(row['chl']), 3) if 'chl' in row and not pd.isna(row['chl']) else None

    reporte_rows.append({
        'rank':           i + 1,
        'semaforo':       get_semaforo(score),
        'score':          round(score, 3),
        'lat_base':       round(float(row['LAT_REF']), 4),
        'lon_base':       round(float(row['LON_REF']), 4),
        'lat_T8':         round(float(row['LAT_T8']),  4),
        'lon_T8':         round(float(row['LON_T8']),  4),
        'lat_T16':        round(float(row['LAT_T16']), 4),
        'lon_T16':        round(float(row['LON_T16']), 4),
        'lat_T24':        round(float(row['LAT_T24']), 4),
        'lon_T24':        round(float(row['LON_T24']), 4),
        'dist_km':        round(float(row['dist_ch']),  1),
        'desp_km':        round(float(row['delta_km']), 1),
        'dlat_por_hora':  round(dlat_por_hora, 6),
        'dlon_por_hora':  round(dlon_por_hora, 6),
        'sst':            sst_val,
        'chl':            chl_val,
        'fecha':          FECHA_FIN_STR,
        'chl_fuente':     'VIIRS+CMEMS' if VIIRS_DISPONIBLE else 'CMEMS',
        'ekman_fuente':   'ERA5'
    })

df_rep = pd.DataFrame(reporte_rows)

try:
    ws_r = sh.worksheet('reporte_diario')
    ws_r.clear()
except:
    ws_r = sh.add_worksheet(title='reporte_diario', rows=150, cols=25)

set_with_dataframe(ws_r, df_rep)
print(f"Reporte exportado: {len(df_rep)} puntos")
sys.stdout.flush()

df_export = df_final.nlargest(5000, 'score_fusionado')[
    ['LAT_REF','LON_REF','score_fusionado',
     'LAT_T8','LON_T8','LAT_T16','LON_T16',
     'LAT_T24','LON_T24','delta_km','dist_ch','fecha']
].reset_index(drop=True)

try:
    ws = sh.worksheet('escala3_v62')
    ws.clear()
except:
    ws = sh.add_worksheet(title='escala3_v62', rows=5100, cols=15)

set_with_dataframe(ws, df_export)
print(f"Mapa exportado: {len(df_export)} filas")

print("\n" + "="*60)
print(f"PredictaMAR v6.2 COMPLETO — {FECHA_FIN_STR}")
print(f"Fin: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("="*60)
sys.stdout.flush()
