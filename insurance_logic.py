import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RIGMATA_XLSX = os.path.join(BASE_DIR, "data", "All Faults new cord 000.xlsx")
KATASTASH_PINAKAS_XLSX = os.path.join(BASE_DIR, "data", "pinakas katastasis spitiou.xlsx")

CORINE_SHAPEFILE_DIR = os.path.join(BASE_DIR, "gis", "Corine")
FIRE_XLSX = os.path.join(BASE_DIR, "data", "Fwtia Jup new.xlsx")

AREA_INFO_SHP = os.path.join(BASE_DIR, "gis", "Areas", "00 Area with Info.shp")
LAKES_SHP = os.path.join(BASE_DIR, "gis", "LAKES", "LAKES.shp")
RIVERS_SHP = os.path.join(BASE_DIR, "gis", "RIVERS", "ALL rivers.shp")
SEA_SHP = os.path.join(BASE_DIR, "gis", "SEA", "Sea Vert.shp")
DEM_TIF_FOR_FLOOD = os.path.join(BASE_DIR, "gis", "Flood", "GR_DEM_COP30.tif")
SLOPE_SHP = os.path.join(BASE_DIR, "gis", "Flood", "GR_Slope.shp")

WIND_SHP = os.path.join(BASE_DIR, "gis", "Ionio", "ionio and others.shp")
SNOW_DEM_TIF = os.path.join(BASE_DIR, "gis", "Height", "GR_DEM_COP30.tif")

POSTCODES_XLSX = os.path.join(BASE_DIR, "data", "PostCodes Greece.xlsx")
POSTAL_CODES_SHP = os.path.join(BASE_DIR, "gis", "Postal Codes", "TK Codes.shp")
