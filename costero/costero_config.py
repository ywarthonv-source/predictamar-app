# ================================================================
# PredictaMAR Costero v1.0 -- CONFIGURACION CENTRAL
# Puerto Chorrillos - 0-20 km - Sistema Corriente de Humboldt
# ================================================================

# -- Area geografica ----------------------------------------------
LAT_CHORRILLOS   = -12.157
LON_CHORRILLOS   = -77.021

# Bounding box 20 km desde el puerto
LAT_MIN_C        = -12.337
LAT_MAX_C        = -11.977
LON_MIN_C        = -77.201
LON_MAX_C        = -76.841

# -- Resolucion y grilla ------------------------------------------
RESOLUCION_M     = 10        # grilla comun en metros
RADIO_MAX_KM     = 20        # radio maximo de operacion
RADIO_ORILLA_KM  = 0.5       # zona de exclusion costera

# -- Buffer pixel mixto -------------------------------------------
BUFFER_LANDSAT_M = 100       # mascara terrestre Landsat
BUFFER_S2_M      = 30        # mascara terrestre Sentinel-2

# -- Umbrales de semaforo -----------------------------------------
UMBRAL_VERDE     = 0.62
UMBRAL_AMARILLO  = 0.52

# -- Lag clorofila ------------------------------------------------
LAG_CHL_DIAS     = 4         # dias de desfase bloom -> cardumen

# -- IPO diferenciado por grupo ecologico -------------------------
IPO_DIAS_PELAGICOS   = 3     # pejerrey, lisa
IPO_DIAS_DEMERSALES  = 5     # cachema, lorna, chauchilla

# -- Decaimiento temporal Fc --------------------------------------
# Fc = max(0, 1 - dias_antiguedad / dias_maximos)
FC_ALOS2_MAX_DIAS    = 14
FC_S2_MAX_DIAS       = 5
FC_S1_MAX_DIAS       = 3

# -- Pesos MicroScore -- modo normal (con Sentinel-2) -------------
W_NORMAL = {
    "s1_sar":    0.35,   # rugosidad superficial
    "s2_optico": 0.30,   # turbidez / clorofila
    "alos2":     0.15,   # estructura fondo x Fc
    "gebco":     0.20,   # batimetria estatica
}

# -- Pesos MicroScore -- modo degradado (sin Sentinel-2) ----------
W_DEGRADADO = {
    "s1_sar":    0.50,
    "s2_optico": 0.00,
    "alos2":     0.25,
    "gebco":     0.25,
}

# -- Umbrales de confianza ----------------------------------------
CONFIANZA_ALTA   = 0.75      # S1 fresco + S2 fresco
CONFIANZA_MEDIA  = 0.50      # S1 fresco, S2 degradado
CONFIANZA_BAJA   = 0.25      # solo climatologico + batimetria

# -- Historial ----------------------------------------------------
HISTORIAL_DIAS   = 90        # dias de historial a conservar

# -- Scheduler ----------------------------------------------------
# GitHub Actions corre a las 08:00 UTC = 03:00 AM Lima
# y a las 20:00 UTC = 03:00 PM Lima
HORA_UTC_MANANA  = "08:00"
HORA_UTC_TARDE   = "20:00"

# -- CMEMS productos ----------------------------------------------
CMEMS_SST_L4     = "cmems_obs-sst_glo_phy_nrt_l4_P1D-m"
CMEMS_CUR        = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"

# -- GEE colecciones ----------------------------------------------
GEE_S1           = "COPERNICUS/S1_GRD"
GEE_S2           = "COPERNICUS/S2_SR_HARMONIZED"
GEE_ALOS2        = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
GEE_GEBCO        = "NOAA/NGDC/ETOPO1"
GEE_GEBCO_BAND   = "bedrock"  # banda de batimetria en ETOPO1

# -- Nombre del sistema -------------------------------------------
VERSION          = "v1.0"
NOMBRE_SISTEMA   = "PredictaMAR Costero v1.0"
PUERTO_ORIGEN    = "Puerto Chorrillos"
