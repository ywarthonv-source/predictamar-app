# ================================================================
# PredictaMAR Costero v1.1 -- CONFIGURACION CENTRAL
# Puerto Chorrillos - 0-20 km - Sistema Corriente de Humboldt
# ================================================================

# -- Area geografica
LAT_CHORRILLOS   = -12.157
LON_CHORRILLOS   = -77.021
# AOI oceanico: solo mar al oeste de la costa de Chorrillos
# LON_MAX cerca de la costa para no incluir tierra
# LON_MIN 20km al oeste para cubrir mar abierto
LAT_MIN_C        = -12.357
LAT_MAX_C        = -11.957
LON_MIN_C        = -77.241
LON_MAX_C        = -77.060  # corregido: Costa Verde Miraflores en -77.050, margen seguridad

# -- Resolucion y grilla
RADIO_MAX_KM     = 6    # radio real operacional de Christian con motor 15HP
RADIO_COSTERO_KM  = 6    # sector costero cubre zona completa de Christian
RADIO_ORILLA_KM  = 0.5

# -- Buffer pixel mixto
BUFFER_LANDSAT_M = 100
BUFFER_S2_M      = 30

# -- Umbrales semaforo
UMBRAL_VERDE     = 0.62
UMBRAL_AMARILLO  = 0.52

# -- Cadena trofica temporal (Bertrand et al. 2008, HCS)
LAG_CHL_DIAS     = 4    # bloom -> cardumen
T_SURGENCIA_DIAS = 7    # surgencia -> bloom

# -- IPO diferenciado
IPO_DIAS_PELAGICOS  = 3   # pejerrey, lisa
IPO_DIAS_DEMERSALES = 5   # cachema, lorna, chauchilla

# -- Decaimiento temporal Fc
FC_ALOS2_MAX_DIAS = 14
FC_S2_MAX_DIAS    = 5
FC_S1_MAX_DIAS    = 6   # SAR util hasta 6 dias -- degradacion gradual

# -- Adveccion
# Adveccion calculada desde hora de corrida del pipeline hasta hora de lance
# HORA_LANCE_LIMA se puede pasar como variable de entorno
# Default: 17 (5PM Lima) -- hora punta identificada en campo (segunda salida 4jun2026)
HORA_LANCE_DEFAULT = 17  # 5PM Lima -- hora punta cachema segun Christian


# -- Oleaje: flag operacional y penalizacion mezcla
SWH_KILL_SWITCH  = 1.5   # metros -- bloqueo operacional
SWH_MEZCLA_MAX   = 2.5   # metros -- mezcla total, penalizacion maxima

# -- PESOS BASE (suman 1.0, redistribuibles si fuente falla)
# Cadena trofica temporal
W_SST_GRAD    = 0.18   # surgencia T-7d SST L4 CMEMS
W_ERA5_VIENTO = 0.12   # viento ERA5 proxy surgencia
W_CHL_BLOOM   = 0.15   # bloom T-4d Sentinel-2
# Condiciones fisicas actuales
W_S1_SAR      = 0.15   # rugosidad SAR Sentinel-1
W_ALOS2       = 0.10   # estructura fondo ALOS-2 banda L
W_BATIMETRIA  = 0.15   # pendiente fondo ETOPO1
W_GEOMETRIA   = 0.15   # gradiente perp. batimetrico (retenciones)

W_BASE = {
    "sst_grad":    W_SST_GRAD,
    "era5_viento": W_ERA5_VIENTO,
    "chl_bloom":   W_CHL_BLOOM,
    "s1_sar":      W_S1_SAR,
    "alos2":       W_ALOS2,
    "batimetria":  W_BATIMETRIA,
    "geometria":   W_GEOMETRIA,
}

# -- Corrientes y oleaje actuan fuera del score base
# Corrientes -> adveccion de posicion proyectada
# Oleaje     -> multiplicador inverso + kill-switch

# -- CMEMS datasets verificados 2025
CMEMS_SST_L4         = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
CMEMS_CORRIENTES_NRT = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"

# -- ERA5 via CDS
ERA5_DATASET   = "reanalysis-era5-single-levels"
ERA5_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind"]

# -- GEE colecciones
GEE_S1        = "COPERNICUS/S1_GRD"
GEE_S2        = "COPERNICUS/S2_SR_HARMONIZED"
GEE_ALOS2     = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
GEE_GEBCO     = "NOAA/NGDC/ETOPO1"
GEE_GEBCO_BAND = "bedrock"
GEE_WAVES = "COPERNICUS/MARINE/WAV/ANFC_0_083DEG_PT3H"

# -- Separacion minima entre puntos (distancia entre compaeros artesanales en Chorrillos)
SEP_VARIANTES_KM  = 1.0   # 1km -- distancia que mantienen compaeros entre embarcaciones
SEP_TRAMPAS_KM    = 1.0   # 1km -- igual para trampas estaticas

# -- Umbrales confianza
CONFIANZA_ALTA  = 0.75
CONFIANZA_MEDIA = 0.50
CONFIANZA_BAJA  = 0.25

# -- Historial
HISTORIAL_DIAS = 90

# -- Perfil operacional Christian (primer piloto)
CHRISTIAN_RADIO_KM     = 6.0      # radio real validado en campo -- segunda salida 4jun2026
CHRISTIAN_PROF_MAX_M   = 3.5      # altura maxima de red
CHRISTIAN_ESPECIES     = ["cachema", "lorna", "pejerrey", "machete", "sardina"]

# -- Capa empirica Christian v1.0 (primer punto validado en campo 27 mayo 2026)
# Formato: (lat, lon, especie_principal, condicion_activacion, bonus)
# condicion_activacion: surgencia minima para activar el bonus
CHRISTIAN_ZONAS_EMPIRICAS = [
    {
        "lat": -12.1338782,
        "lon": -77.0586627,
        "nombre": "Zona principal Christian -- validada 27may2026",
        "especie": "cachema",
        "surgencia_min": 0.60,   # solo activa si surgencia >= 60%
        "bonus": 0.20,           # bonus +20% cuando condicion activa
        "hora_optima": "17:30",  # hora de mejor captura observada
        "notas": "Agua lechosa turbia, lobos marinos activos, ~2 baldes cachema"
    }
]

# -- Segunda zona empirica validada en campo 4 junio 2026
# Punto donde Christian lanzo la red a las 5:57PM -- pesca positiva con lobos
CHRISTIAN_ZONAS_EMPIRICAS.append({
    "lat": -12.1612,
    "lon": -77.0626,
    "nombre": "Zona segunda salida -- validada 04jun2026",
    "especie": "cachema",
    "surgencia_min": 0.60,
    "bonus": 0.15,           # bonus menor que zona principal (menos datos)
    "hora_optima": "18:00",  # hora de lance real: 5:57PM
    "notas": "Agua clara, corriente Norte, lobos activos, pesca positiva reducida por lobos"
})

# -- Zonas de exclusion por fondo rocoso (red se rompe)
# A completar con Christian en proxima sesion
CHRISTIAN_ZONAS_ROCOSAS = []  # pendiente: mapeo con Christian

# -- NASA Earthdata credentials (MUR SST 1km)
# Agregar en GitHub Secrets: NASA_EARTHDATA_USER y NASA_EARTHDATA_PASS
# Usuario registrado: randywarthonvelarde (ywarthonv@gmail.com)
# URL: urs.earthdata.nasa.gov

# -- Version
VERSION        = "v1.2"
NOMBRE_SISTEMA = "PredictaMAR Costero v1.2"
PUERTO_ORIGEN  = "Puerto Chorrillos"
