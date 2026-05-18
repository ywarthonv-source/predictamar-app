# ================================================================
# CELDA 13 — Exportar a Google Sheets y Drive
# ================================================================

# --- Recalcular desplazamientos ---
def desp(uo, vo, h, lat=-12.0):
    s = h * 3600
    return (vo*s)/111000, (uo*s)/(111000*np.cos(np.radians(lat)))

lat_med   = df_final['LAT_REF'].mean()
dl8,  dn8  = desp(uo_mean_val, vo_mean_val, 8,  lat_med)
dl16, dn16 = desp(uo_mean_val, vo_mean_val, 16, lat_med)
dl24, dn24 = desp(uo_mean_val, vo_mean_val, 24, lat_med)

dlat_por_hora = dl16 / 16
dlon_por_hora = dn16 / 16

print(f"Desplazamiento T-16h: {np.sqrt((dl16*111)**2+(dn16*111)**2):.2f} km")
print(f"dlat/h: {dlat_por_hora:.6f} | dlon/h: {dlon_por_hora:.6f}")

# --- Zonas A y B ---
df_final['LAT_T8']  = df_final['LAT_REF'] + dl8
df_final['LON_T8']  = df_final['LON_REF'] + dn8
df_final['LAT_T16'] = df_final['LAT_REF'] + dl16
df_final['LON_T16'] = df_final['LON_REF'] + dn16
df_final['LAT_T24'] = df_final['LAT_REF'] + dl24
df_final['LON_T24'] = df_final['LON_REF'] + dn24
df_final['delta_km'] = np.sqrt((dl16*111)**2 + (dn16*111)**2)

def distancia_km_fn(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111
    dlon = (lon2 - lon1) * 111 * np.cos(np.radians((lat1+lat2)/2))
    return np.sqrt(dlat**2 + dlon**2)

LAT_CHRISTIAN = -12.15
LON_CHRISTIAN = -77.02

df_final['dist_ch'] = df_final.apply(
    lambda r: distancia_km_fn(LAT_CHRISTIAN, LON_CHRISTIAN,
                               r['LAT_T16'], r['LON_T16']), axis=1
)

zona_a = df_final[df_final['dist_ch'] <= 40].copy()
zona_b = df_final[
    (df_final['dist_ch'] > 40) &
    (df_final['dist_ch'] <= 80)
].copy()

print(f"Zona A: {len(zona_a)} puntos | Zona B: {len(zona_b)} puntos")

# --- Reporte diario completo para Streamlit ---
reporte_rows = []

for zona_nom, zona_df in [("A_15HP", zona_a), ("B_40HP", zona_b)]:
    if len(zona_df) == 0:
        continue
    for i, row in zona_df.nlargest(3, 'score_fusionado').reset_index(drop=True).iterrows():
        score    = row['score_fusionado']
        semaforo = "VERDE"    if score >= 0.62 else \
                   "AMARILLO" if score >= 0.52 else "ROJO"

        # SST y CHL del punto mas cercano en df_grilla
        try:
            idx_cercano = ((df_grilla['LAT_REF'] - row['LAT_REF'])**2 +
                           (df_grilla['LON_REF'] - row['LON_REF'])**2).idxmin()
            punto_c = df_grilla.iloc[idx_cercano]
            sst_val = round(float(punto_c['sst']), 2) if 'sst' in df_grilla.columns else None
            chl_val = round(float(punto_c['chl']), 3) if 'chl' in df_grilla.columns else None
        except:
            sst_val = None
            chl_val = None

        reporte_rows.append({
            'zona':           zona_nom,
            'rank':           i + 1,
            'semaforo':       semaforo,
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
            'ekman_fuente':   'ERA5' if ERA5_DISPONIBLE else 'proxy'
        })

df_rep = pd.DataFrame(reporte_rows)

# --- Exportar reporte a Sheets ---
try:
    ws_r = sh.worksheet('reporte_diario')
    ws_r.clear()
except:
    ws_r = sh.add_worksheet(title='reporte_diario', rows=100, cols=25)

set_with_dataframe(ws_r, df_rep)
print(f"\nReporte exportado: {len(df_rep)} zonas")
print(f"Columnas: {list(df_rep.columns)}")

# --- Exportar mapa top 5000 a Sheets ---
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

# --- Drive ---
df_final.to_parquet(
    f"{DRIVE_BASE}/raw/predictamar_v62_{FECHA_FIN_STR}.parquet",
    index=False
)
print(f"Drive guardado: predictamar_v62_{FECHA_FIN_STR}.parquet")
print("\nPredictaMAR v6.2 FINAL completo.")
