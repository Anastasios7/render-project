import os
import re
import math
import time
import warnings
import unicodedata

from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pyproj

from shapely.geometry import Point, Polygon
from shapely.ops import transform

from pyproj import Transformer
from rasterio.transform import rowcol
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
from scipy.stats import norm


# ==========================================
# 1. SETTINGS / CONSTANTS
# ==========================================
VERBOSE = False

THEFT_DAMAGE_RATIO = 0.015
EARTHQUAKE_DEDUCTIBLE = 0.02
NOW_YEAR = datetime.now().year

FLOOD_DAMAGE_RATIO = 0.03
FLOOD_GROUP = 3
FLOOD_OTHER_AREA = 66641.118

a1_noa = 3
a2_ypen_sev = 10
a3_ypen_norm = 3
a4_paper_points = 10

year1_noa = 30
year2_ypen_sev = 30
year3_ypen_norm = 30
year4_paper_points = 100


# ==========================================
# 2. PATHS
# ==========================================
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


# ==========================================
# 3. WARNINGS
# ==========================================
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")


# ==========================================
# 4. HELPERS
# ==========================================
def dprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def split(self):
        return time.perf_counter() - self.t0


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


# ==========================================
# 5. GEOCODER HELPERS
# ==========================================
def geo_remove_tonos(geo_text):
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(geo_text))
        if unicodedata.category(ch) != "Mn"
    )


def geo_normalize_text(geo_text):
    geo_text = str(geo_text).strip()
    geo_text = re.sub(r"\s+", " ", geo_text)
    return geo_text


def geo_normalize_for_compare(geo_text):
    geo_text = geo_normalize_text(geo_text).lower()
    geo_text = geo_remove_tonos(geo_text)
    geo_text = re.sub(r"[^a-z0-9α-ω\s]", " ", geo_text)
    geo_text = re.sub(r"\s+", " ", geo_text).strip()
    return geo_text


def geo_clean_postcode(geo_postcode):
    return re.sub(r"\D", "", str(geo_postcode)).strip()


def geo_similarity(geo_a, geo_b):
    return SequenceMatcher(
        None,
        geo_normalize_for_compare(geo_a),
        geo_normalize_for_compare(geo_b)
    ).ratio()


def geo_safe_get(geo_dict, geo_key, geo_default=""):
    geo_val = geo_dict.get(geo_key, geo_default)
    return "" if geo_val is None else str(geo_val)


def geo_country_code(country_name):
    country_name = geo_normalize_for_compare(country_name)
    if country_name in ["ελλαδα", "greece", "gr"]:
        return "gr"
    if country_name in ["κυπρος", "cyprus", "cy"]:
        return "cy"
    return None


def geo_build_address_candidates(geo_street, geo_number, geo_city, geo_postcode, geo_country="Ελλάδα"):
    geo_street = geo_normalize_text(geo_street)
    geo_number = geo_normalize_text(geo_number)
    geo_city = geo_normalize_text(geo_city)
    geo_postcode = geo_clean_postcode(geo_postcode)
    geo_country = geo_normalize_text(geo_country)

    geo_candidates = []

    if geo_street and geo_number and geo_city and geo_postcode:
        geo_candidates.append({
            "label": "Πλήρης διεύθυνση",
            "query": f"{geo_street} {geo_number}, {geo_city}, {geo_postcode}, {geo_country}"
        })

    if geo_street and geo_number and geo_city and geo_postcode:
        geo_candidates.append({
            "label": "Πλήρης χωρίς κόμματα",
            "query": f"{geo_street} {geo_number} {geo_city} {geo_postcode} {geo_country}"
        })

    if geo_street and geo_city and geo_postcode:
        geo_candidates.append({
            "label": "Χωρίς αριθμό",
            "query": f"{geo_street}, {geo_city}, {geo_postcode}, {geo_country}"
        })

    if geo_street and geo_number and geo_city:
        geo_candidates.append({
            "label": "Χωρίς ΤΚ",
            "query": f"{geo_street} {geo_number}, {geo_city}, {geo_country}"
        })

    if geo_street and geo_city:
        geo_candidates.append({
            "label": "Οδός + πόλη",
            "query": f"{geo_street}, {geo_city}, {geo_country}"
        })

    if geo_city and geo_postcode:
        geo_candidates.append({
            "label": "Πόλη + ΤΚ",
            "query": f"{geo_city}, {geo_postcode}, {geo_country}"
        })

    if geo_city:
        geo_candidates.append({
            "label": "Μόνο πόλη",
            "query": f"{geo_city}, {geo_country}"
        })

    geo_seen = set()
    geo_unique = []
    for geo_item in geo_candidates:
        if geo_item["query"] not in geo_seen:
            geo_seen.add(geo_item["query"])
            geo_unique.append(geo_item)

    return geo_unique


def geo_extract_address_parts(geo_location):
    geo_raw_addr = {}
    try:
        geo_raw_addr = geo_location.raw.get("address", {})
    except Exception:
        geo_raw_addr = {}

    geo_road = (
        geo_safe_get(geo_raw_addr, "road")
        or geo_safe_get(geo_raw_addr, "pedestrian")
        or geo_safe_get(geo_raw_addr, "footway")
        or geo_safe_get(geo_raw_addr, "residential")
    )

    geo_house_number = geo_safe_get(geo_raw_addr, "house_number")

    geo_city = (
        geo_safe_get(geo_raw_addr, "city")
        or geo_safe_get(geo_raw_addr, "town")
        or geo_safe_get(geo_raw_addr, "village")
        or geo_safe_get(geo_raw_addr, "municipality")
        or geo_safe_get(geo_raw_addr, "suburb")
    )

    geo_postcode = geo_safe_get(geo_raw_addr, "postcode")
    geo_country = geo_safe_get(geo_raw_addr, "country")

    return {
        "road": geo_road,
        "house_number": geo_house_number,
        "city": geo_city,
        "postcode": geo_postcode,
        "country": geo_country,
    }


def geo_compare_and_describe(geo_user_street, geo_user_number, geo_user_city, geo_user_postcode, geo_matched_parts):
    geo_messages = []

    geo_matched_road = geo_matched_parts["road"]
    geo_matched_number = geo_matched_parts["house_number"]
    geo_matched_city = geo_matched_parts["city"]
    geo_matched_postcode = geo_matched_parts["postcode"]

    if geo_matched_road:
        geo_s = geo_similarity(geo_user_street, geo_matched_road)
        if geo_normalize_for_compare(geo_user_street) != geo_normalize_for_compare(geo_matched_road):
            if geo_s >= 0.75:
                geo_messages.append(
                    f"Οδός: δόθηκε '{geo_user_street}', βρέθηκε/διορθώθηκε σε '{geo_matched_road}'."
                )
            else:
                geo_messages.append(
                    f"Οδός: δόθηκε '{geo_user_street}', αλλά το αποτέλεσμα αντιστοιχίστηκε σε '{geo_matched_road}'."
                )
        else:
            geo_messages.append(f"Οδός: '{geo_user_street}' → ΟΚ.")
    else:
        geo_messages.append("Οδός: δεν επιβεβαιώθηκε από το geocoder.")

    geo_user_number_clean = re.sub(r"\D", "", str(geo_user_number))
    geo_matched_number_clean = re.sub(r"\D", "", str(geo_matched_number))

    if geo_matched_number:
        if geo_user_number_clean != geo_matched_number_clean:
            geo_messages.append(
                f"Αριθμός: δόθηκε '{geo_user_number}', βρέθηκε '{geo_matched_number}'."
            )
        else:
            geo_messages.append(f"Αριθμός: '{geo_user_number}' → ΟΚ.")
    else:
        geo_messages.append("Αριθμός: δεν επιβεβαιώθηκε από το geocoder.")

    if geo_matched_city:
        if geo_normalize_for_compare(geo_user_city) != geo_normalize_for_compare(geo_matched_city):
            geo_messages.append(
                f"Πόλη: δόθηκε '{geo_user_city}', βρέθηκε/διορθώθηκε σε '{geo_matched_city}'."
            )
        else:
            geo_messages.append(f"Πόλη: '{geo_user_city}' → ΟΚ.")
    else:
        geo_messages.append("Πόλη: δεν επιβεβαιώθηκε από το geocoder.")

    geo_user_pc = geo_clean_postcode(geo_user_postcode)
    geo_matched_pc = geo_clean_postcode(geo_matched_postcode)

    if geo_matched_postcode:
        if geo_user_pc != geo_matched_pc:
            geo_messages.append(
                f"ΤΚ: δόθηκε '{geo_user_postcode}', βρέθηκε '{geo_matched_postcode}'."
            )
        else:
            geo_messages.append(f"ΤΚ: '{geo_user_postcode}' → ΟΚ.")
    else:
        geo_messages.append("ΤΚ: δεν επιβεβαιώθηκε από το geocoder.")

    return geo_messages


def geo_address_to_lonlat_with_report(
    geo_street,
    geo_number,
    geo_city,
    geo_postcode,
    geo_country="Ελλάδα",
    geo_verbose=False
):
    geo_geolocator = Nominatim(user_agent="insurance_geocoder_app", timeout=10)

    geo_candidates = geo_build_address_candidates(
        geo_street, geo_number, geo_city, geo_postcode, geo_country
    )

    geo_tried = []
    country_code = geo_country_code(geo_country)

    for geo_item in geo_candidates:
        geo_query = geo_item["query"]
        geo_label = geo_item["label"]

        try:
            kwargs = {
                "query": geo_query,
                "exactly_one": True,
                "addressdetails": True
            }
            if country_code:
                kwargs["country_codes"] = country_code

            geo_location = geo_geolocator.geocode(**kwargs)

            geo_tried.append(f"[{geo_label}] {geo_query}")

            if geo_location:
                geo_lon = float(geo_location.longitude)
                geo_lat = float(geo_location.latitude)

                geo_matched_parts = geo_extract_address_parts(geo_location)
                geo_corrections = geo_compare_and_describe(
                    geo_user_street=geo_street,
                    geo_user_number=geo_number,
                    geo_user_city=geo_city,
                    geo_user_postcode=geo_postcode,
                    geo_matched_parts=geo_matched_parts
                )

                geo_result = {
                    "longitude": geo_lon,
                    "latitude": geo_lat,
                    "matched_address": geo_location.address,
                    "matched_parts": geo_matched_parts,
                    "used_query": geo_query,
                    "used_query_label": geo_label,
                    "tried_queries": geo_tried,
                    "corrections": geo_corrections,
                }

                if geo_verbose:
                    print("========================================")
                    print("ΒΡΕΘΗΚΕ ΑΝΤΙΣΤΟΙΧΙΣΗ")
                    print("========================================")
                    print("Query type      :", geo_label)
                    print("Query used      :", geo_query)
                    print("Matched address :", geo_location.address)
                    print("LATITUDE        :", geo_lat)
                    print("LONGITUDE       :", geo_lon)
                    print("========================================")

                return geo_result

        except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
            geo_tried.append(f"[{geo_label}] {geo_query}  --> Geocoder error: {geo_err}")
            continue
        except Exception as geo_err:
            geo_tried.append(f"[{geo_label}] {geo_query}  --> Unknown error: {geo_err}")
            continue

    raise ValueError(
        "Δεν βρέθηκαν συντεταγμένες.\n"
        "Queries που δοκιμάστηκαν:\n- " + "\n- ".join(geo_tried)
    )


# ==========================================
# 6. POSTCODE FROM SHAPEFILE
# ==========================================
_POSTCODES_GDF = None

def load_postcodes_gdf():
    global _POSTCODES_GDF
    if _POSTCODES_GDF is not None:
        return _POSTCODES_GDF

    gdf = gpd.read_file(POSTAL_CODES_SHP)
    gdf.columns = [c.strip() for c in gdf.columns]

    if "POSTCODE" not in gdf.columns:
        raise KeyError(f"Δεν υπάρχει στήλη 'POSTCODE'. Στήλες: {list(gdf.columns)}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    _ = gdf.sindex
    _POSTCODES_GDF = gdf
    return gdf


def postcode_from_lonlat_int(lon, lat, fallback_value=11):
    gdf = load_postcodes_gdf()
    pt = Point(lon, lat)

    hit = gdf[gdf.geometry.covers(pt)]
    if hit.empty:
        print(f"Longitude: {lon} | Latitude: {lat} -> Δεν βρήκε TK, βάζω {fallback_value}")
        return int(fallback_value)

    val = hit.iloc[0]["POSTCODE"]
    digits = "".join(ch for ch in str(val) if ch.isdigit())

    if not digits:
        print(f"Longitude: {lon} | Latitude: {lat} -> Δεν βρήκε TK (κενό/λάθος POSTCODE), βάζω {fallback_value}")
        return int(fallback_value)

    try:
        return int(digits)
    except Exception:
        print(f"Longitude: {lon} | Latitude: {lat} -> Δεν βρήκε TK (δεν γίνεται int: {val}), βάζω {fallback_value}")
        return int(fallback_value)


# ==========================================
# 7. EARTHQUAKE
# ==========================================
def load_earthquake_data():
    rigmata = pd.read_excel(RIGMATA_XLSX)
    katastashpinakas = pd.read_excel(
        KATASTASH_PINAKAS_XLSX,
        index_col=0,
        dtype={'DS1': float, 'DS2': float, 'DS3': float, 'DS4': float, 'DS5': float, 'β': float}
    )
    return rigmata, katastashpinakas


def damages(PGA, tabkat, katastash_p, apalagi):
    bb = tabkat.loc[katastash_p, 'β']
    PDS1 = norm.cdf((1 / bb) * np.log(PGA / tabkat.loc[katastash_p, 'DS1']))
    PDS2 = norm.cdf((1 / bb) * np.log(PGA / tabkat.loc[katastash_p, 'DS2']))
    PDS3 = norm.cdf((1 / bb) * np.log(PGA / tabkat.loc[katastash_p, 'DS3']))
    PDS4 = norm.cdf((1 / bb) * np.log(PGA / tabkat.loc[katastash_p, 'DS4']))
    PDS5 = norm.cdf((1 / bb) * np.log(PGA / tabkat.loc[katastash_p, 'DS5']))

    P_00_01 = PDS1 - PDS2
    P_01_10 = PDS2 - PDS3
    P_10_30 = PDS3 - PDS4
    P_30_60 = PDS4 - PDS5
    P_60_100 = PDS5

    maxval = (
        P_00_01 * max(0, 0.01 - apalagi) +
        P_01_10 * max(0, 0.1 - apalagi) +
        P_10_30 * max(0, 0.3 - apalagi) +
        P_30_60 * max(0, 0.6 - apalagi) +
        P_60_100 * max(0, 1.0 - apalagi)
    )
    midval = (
        P_00_01 * max(0, 0.005 - apalagi) +
        P_01_10 * max(0, 0.045 - apalagi) +
        P_10_30 * max(0, 0.2 - apalagi) +
        P_30_60 * max(0, 0.45 - apalagi) +
        P_60_100 * max(0, 0.8 - apalagi)
    )
    minval = (
        P_00_01 * max(0, 0.0 - apalagi) +
        P_01_10 * max(0, 0.01 - apalagi) +
        P_10_30 * max(0, 0.1 - apalagi) +
        P_30_60 * max(0, 0.3 - apalagi) +
        P_60_100 * max(0, 0.6 - apalagi)
    )
    return minval, midval, maxval


def katastashdef(year, floors, number, material, noww):
    year = int(year)
    floors = int(floors)
    number = str(number)
    material = str(material)

    if year < 1959:
        code = 'P'
    elif year < 1985:
        code = 'L'
    elif year < 1999:
        code = 'M'
    elif year < noww:
        code = 'H'
    else:
        raise ValueError(f"Bad data Year: {year}")

    if 0 < floors <= 2:
        h = 'L'
    elif floors <= 5:
        h = 'M'
    elif floors > 5:
        h = 'H'
    else:
        raise ValueError(f"Bad data Floors: {floors}")

    if number not in ['1', '3.1', '3.2']:
        raise ValueError(f"Bad data Number: {number}")

    if code == 'P':
        kat = 'RC' + '1' + h + 'L'
        extra = 1
    elif code == 'H':
        kat = 'RC' + '1' + h + 'M'
        extra = 2
    else:
        kat = 'RC' + number + h + code
        extra = 0

    if material == 'Concrete':
        extra_new = 0
    elif material == 'Wood':
        kat = 'RC' + number + h + code
        extra_new = 3
    elif material == 'Metal':
        kat = 'RC' + number + h + code
        extra_new = 4
    else:
        raise ValueError(f"Bad data Material: {material}")

    return kat, extra, extra_new


def apostasi(df_faults, lon, lat, year, floors, number, material, katastashpinakas, apalagi, noww):
    ss = len(df_faults)
    result_table = np.zeros((ss, 11), dtype=object)

    point = Point(lon, lat)
    katastash, ex, ex_n = katastashdef(year, floors, number, material, noww)

    for i in range(ss):
        polygon_points = []
        for j in range(1, ((len(df_faults.columns) - 3) // 2) + 1):
            x_col = f'X_{j}'
            y_col = f'Y_{j}'
            if x_col in df_faults.columns and y_col in df_faults.columns:
                if pd.notnull(df_faults[x_col].iloc[i]) and pd.notnull(df_faults[y_col].iloc[i]):
                    polygon_points.append((df_faults[x_col].iloc[i], df_faults[y_col].iloc[i]))

        if len(polygon_points) < 3:
            continue

        polygon = Polygon(polygon_points)

        if polygon.contains(point):
            distance = 0
        else:
            nearest_point = polygon.exterior.interpolate(polygon.exterior.project(point))
            nearest_point_coords = (nearest_point.y, nearest_point.x)
            distance = geodesic((lat, lon), nearest_point_coords).kilometers

        PGA = np.exp(0.82 * df_faults['maxmax'].iloc[i] - 1.59 * np.log(distance + 15) + 5.25) / 981

        result_table[i][0] = PGA
        result_table[i][1] = distance
        result_table[i][2] = df_faults['maxmax'].iloc[i]
        result_table[i][3] = df_faults['tmean'].iloc[i]
        result_table[i][4] = df_faults['Name'].iloc[i]

        temp1, temp2, temp3 = damages(PGA, katastashpinakas, katastash, apalagi)

        if ex == 1:
            temp1 *= 1.3
            temp2 *= 1.3
            temp3 *= 1.3
        elif ex == 2:
            temp1 *= 0.7
            temp2 *= 0.7
            temp3 *= 0.7

        if ex_n == 3:
            temp1 *= 0.5
            temp2 *= 0.5
            temp3 *= 0.5
        elif ex_n == 4:
            temp1 *= 0.2
            temp2 *= 0.2
            temp3 *= 0.2

        result_table[i][5], result_table[i][6], result_table[i][7] = temp1, temp2, temp3
        result_table[i][8] = temp1 * (1 / df_faults['tmean'].iloc[i])
        result_table[i][9] = temp2 * (1 / df_faults['tmean'].iloc[i])
        result_table[i][10] = temp3 * (1 / df_faults['tmean'].iloc[i])

    sorted_indices = np.argsort(result_table[:, 0])[::-1]
    result_table = result_table[sorted_indices]

    result_df = pd.DataFrame(
        result_table,
        columns=['PGA', 'Distance', 'MaxMag', 'T-mean', 'Name', 'Minval', 'Midval', 'Maxval', 'MinCost', 'MidCost', 'MaxCost']
    )

    cost_min = pd.to_numeric(result_df['MinCost'], errors='coerce').fillna(0).sum()
    cost_mid = pd.to_numeric(result_df['MidCost'], errors='coerce').fillna(0).sum()
    cost_max = pd.to_numeric(result_df['MaxCost'], errors='coerce').fillna(0).sum()

    return result_df, cost_min, cost_mid, cost_max


def earthquake_premium(lon, lat, asset_value, year_construction, floors_house, stability_number, material_house):
    rigmata, katpin = load_earthquake_data()
    _, _, cost_mid, _ = apostasi(
        rigmata,
        lon,
        lat,
        year_construction,
        floors_house,
        stability_number,
        material_house,
        katpin,
        EARTHQUAKE_DEDUCTIBLE,
        NOW_YEAR
    )
    premium = float(cost_mid) * float(asset_value)
    rate = float(cost_mid)
    return rate, premium


# ==========================================
# 8. FIRE
# ==========================================
_FIRE_GDF = None
_FIRE_DF_PROB = None
_FIRE_DF_TK = None

def load_fire_data():
    global _FIRE_GDF, _FIRE_DF_PROB, _FIRE_DF_TK

    if _FIRE_GDF is None:
        gdf = gpd.read_file(CORINE_SHAPEFILE_DIR)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        _FIRE_GDF = gdf

    if _FIRE_DF_PROB is None or _FIRE_DF_TK is None:
        df_prob = pd.read_excel(FIRE_XLSX, sheet_name=0, index_col=0)
        df_tk = pd.read_excel(FIRE_XLSX, sheet_name=1, index_col=0)
        _FIRE_DF_PROB = df_prob
        _FIRE_DF_TK = df_tk


def create_buffer(lon, lat, radius_km):
    local_proj = pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon)
    wgs84_to_local = pyproj.Transformer.from_proj(pyproj.Proj("EPSG:4326"), local_proj, always_xy=True).transform
    local_to_wgs84 = pyproj.Transformer.from_proj(local_proj, pyproj.Proj("EPSG:4326"), always_xy=True).transform

    center = Point(lon, lat)
    center_local = transform(wgs84_to_local, center)
    buffer_local = center_local.buffer(radius_km * 1000)
    buffer_wgs84 = transform(local_to_wgs84, buffer_local)
    return buffer_wgs84


def calculate_area_percentage(circle, gdf):
    intersected = gdf[gdf.geometry.intersects(circle)]
    results = []
    for _, row in intersected.iterrows():
        inter = row.geometry.intersection(circle)
        if not inter.is_empty:
            results.append({"Code_18": row["Code_18"], "Area_in_circle": inter.area})
    total_area = circle.area
    for r in results:
        r["Percentage"] = (r["Area_in_circle"] / total_area) * 100 if total_area else 0
    return results


def get_max_probability(percentages, df_prob):
    max_prob = 0
    for item in percentages:
        code = item["Code_18"]
        try:
            prob = df_prob.loc[int(code), "Probability"]
            if isinstance(prob, pd.Series):
                prob = prob.max()
        except Exception:
            prob = 0
        max_prob = max(max_prob, float(prob))
    return max_prob


def get_weighted_average_probability(percentages, df_prob):
    weighted_sum = 0
    for item in percentages:
        code = item["Code_18"]
        pct = item["Percentage"]
        try:
            prob = df_prob.loc[int(code), "Probability"]
            if isinstance(prob, pd.Series):
                prob = prob.max()
        except Exception:
            prob = 0
        weighted_sum += float(prob) * (pct / 100.0)
    return weighted_sum


def fire_premium(lon, lat, tk, asset_value):
    load_fire_data()
    gdf = _FIRE_GDF
    df_prob = _FIRE_DF_PROB
    df_tk = _FIRE_DF_TK

    radii = {"100m": 0.1, "200m": 0.2, "500m": 0.5}
    results = {}

    for key, radius in radii.items():
        circle = create_buffer(lon, lat, radius)
        percentages = calculate_area_percentage(circle, gdf)
        results[key] = percentages

    P1 = get_max_probability(results["100m"], df_prob)
    P2 = get_max_probability(results["200m"], df_prob)
    P3 = get_weighted_average_probability(results["500m"], df_prob)

    P_all = 0.8 * P1 + 0.1 * P2 + 0.1 * P3

    cost = P_all * df_tk.loc[10, "Damage Ratio"] * asset_value * df_tk.loc[10, "Persentage"]
    premium = float(cost)
    rate = premium / float(asset_value) if asset_value else 0.0
    return rate, premium


# ==========================================
# 9. FLOOD
# ==========================================
_AREA_GDF = None
_AREA_GDF_GEOM = None
_SLOPE_GDF = None

def load_flood_geodata():
    global _AREA_GDF, _AREA_GDF_GEOM, _SLOPE_GDF

    if _AREA_GDF is None:
        _AREA_GDF_GEOM = gpd.read_file(AREA_INFO_SHP)
        _AREA_GDF = _AREA_GDF_GEOM.drop(columns=["geometry"]).copy()

    if _SLOPE_GDF is None:
        _SLOPE_GDF = gpd.read_file(SLOPE_SHP)


def compute_gamma_for_point(lon, lat):
    load_flood_geodata()

    gdf = _AREA_GDF.copy()
    gdf_geom = _AREA_GDF_GEOM

    required = ["paper poin", "ypen_norma", "ypen_sev_d", "NOA"]
    for col in required:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    gdf["All_points"] = gdf["paper poin"] + gdf["ypen_norma"] + gdf["ypen_sev_d"] + gdf["NOA"]
    gdf["fi"] = (
        gdf["paper poin"] * (a4_paper_points / year4_paper_points) +
        gdf["ypen_norma"] * (a3_ypen_norm / year3_ypen_norm) +
        gdf["ypen_sev_d"] * (a2_ypen_sev / year2_ypen_sev) +
        gdf["NOA"] * (a1_noa / year1_noa)
    )

    mask_under = (gdf["All_points"] < FLOOD_GROUP)
    total_final_area_under = gdf.loc[mask_under, "final_area"].sum()
    total_gamma_under = (
        gdf.loc[mask_under, "NOA"].sum() * (a1_noa / year1_noa) +
        gdf.loc[mask_under, "ypen_sev_d"].sum() * (a2_ypen_sev / year2_ypen_sev) +
        gdf.loc[mask_under, "ypen_norma"].sum() * (a3_ypen_norm / year3_ypen_norm) +
        gdf.loc[mask_under, "paper poin"].sum() * (a4_paper_points / year4_paper_points)
    )

    gdf["gamma"] = np.where(
        gdf["All_points"] >= FLOOD_GROUP,
        gdf["fi"] / gdf["final_area"],
        total_gamma_under / (total_final_area_under + FLOOD_OTHER_AREA)
    )

    gdf.loc[gdf["ID_AREA"].isin([196, 142]), "gamma"] = 0.0071

    point = Point(lon, lat)
    match = gdf_geom[gdf_geom.contains(point)]

    if not match.empty:
        area_id = match.iloc[0]["ID_AREA"]
        row = gdf[gdf["ID_AREA"] == area_id]
        gamma = float(row.iloc[0]["gamma"])
    else:
        gamma = float(total_gamma_under / (total_final_area_under + FLOOD_OTHER_AREA))

    if gamma < 0.0031:
        gamma = 0.0031
    elif gamma > 0.02:
        gamma = 0.02

    return gamma


def min_distance_to_water_km(lon, lat):
    point_wgs = Point(lon, lat)
    proj_crs = "EPSG:3857"
    point_proj = gpd.GeoSeries([point_wgs], crs="EPSG:4326").to_crs(proj_crs).iloc[0]

    shapefiles = [LAKES_SHP, RIVERS_SHP, SEA_SHP]
    minwater = None

    for path in shapefiles:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf_proj = gdf.to_crs(proj_crs)

        if gdf_proj.contains(point_proj).any():
            dist_km = 0.0
        else:
            dist_km = float(gdf_proj.geometry.distance(point_proj).min()) / 1000.0

        minwater = dist_km if minwater is None else min(minwater, dist_km)

    return float(minwater if minwater is not None else 0.0)


def elevation_from_dem_for_flood(lon, lat):
    with rasterio.open(DEM_TIF_FOR_FLOOD) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        r, c = rowcol(src.transform, x, y)
        return float(src.read(1)[r, c])


def slope_value(lon, lat):
    load_flood_geodata()
    gdf = _SLOPE_GDF

    point = Point(lon, lat)
    if gdf.crs != "EPSG:4326":
        point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]

    matched = gdf[gdf.contains(point)]
    if matched.empty:
        gdf = gdf.copy()
        gdf["distance"] = gdf.distance(point)
        matched = gdf.loc[[gdf["distance"].idxmin()]]

    raw = matched.iloc[0]["KLISI"]
    try:
        v = float(raw)
        if pd.isna(v):
            v = 1.0
    except Exception:
        v = 1.0

    return float(v)


def flood_probability(dist_km, elev_hm):
    x = dist_km
    y = elev_hm
    return 1 - 0.0644 * x - 0.1324 * y + 0.0009 * x**2 + 0.0072 * x * y + 0.0065 * y**2


def flood_premium(lon, lat, asset_value):
    minwater = min_distance_to_water_km(lon, lat)
    elev_m = elevation_from_dem_for_flood(lon, lat)
    klisi = slope_value(lon, lat)

    prob = float(flood_probability(minwater, elev_m / 100.0))
    if klisi >= 5:
        prob *= 0.5
    elif klisi >= 3:
        prob *= 0.75

    if prob < 0.25:
        prob = 0.25

    gamma = compute_gamma_for_point(lon, lat)

    rate = prob * gamma * FLOOD_DAMAGE_RATIO
    premium = rate * float(asset_value)
    return rate, premium


# ==========================================
# 10. WIND
# ==========================================
_WIND_GDF = None

def load_wind_gdf():
    global _WIND_GDF
    if _WIND_GDF is None:
        gdf = gpd.read_file(WIND_SHP)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4337, allow_override=True)
        _WIND_GDF = gdf


def wind_indicator(lon, lat):
    load_wind_gdf()
    gdf = _WIND_GDF
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf.crs)
    return 1 if gdf.covers(point.iloc[0]).any() else 0


def wind_premium(lon, lat, asset_value):
    rate = wind_indicator(lon, lat) * 0.0001 * 0.1
    premium = rate * float(asset_value)
    return float(rate), float(premium)


# ==========================================
# 11. SNOW
# ==========================================
_SNOW_DS = None
_SNOW_TR = None

def load_snow_dem():
    global _SNOW_DS, _SNOW_TR
    if _SNOW_DS is None:
        _SNOW_DS = rasterio.open(SNOW_DEM_TIF)
        _SNOW_TR = Transformer.from_crs("EPSG:4326", _SNOW_DS.crs, always_xy=True)


def elevation_snow(lon, lat):
    load_snow_dem()
    x, y = _SNOW_TR.transform(lon, lat)
    left, bottom, right, top = _SNOW_DS.bounds

    if not (left <= x <= right and bottom <= y <= top):
        return None

    val = next(_SNOW_DS.sample([(x, y)], indexes=1, masked=True))[0]
    if getattr(val, "mask", False):
        return None

    return float(val)


def snow_premium(lon, lat, asset_value):
    z = elevation_snow(lon, lat)

    if z is None or (isinstance(z, float) and math.isnan(z)):
        rate = 0.0004 * 0.01
    elif z < 100:
        rate = 0.0002 * 0.01
    elif z < 200:
        rate = 0.0003 * 0.01
    elif z < 300:
        rate = 0.0004 * 0.01
    elif z < 600:
        rate = 0.0005 * 0.01
    else:
        rate = 0.0005 * 0.01

    premium = float(rate) * float(asset_value)
    return float(rate), float(premium)


# ==========================================
# 12. THEFT
# ==========================================
_THEFT_DF = None
_THEFT_COLS = {}

def theft_rate(postcode, which="Theft"):
    global _THEFT_DF, _THEFT_COLS

    def _digits(value):
        return "".join(re.findall(r"\d", str(value)))

    def _pick_col(cands, columns):
        norm_col = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        by_norm = {norm_col(c): c for c in columns}

        for cand in cands:
            key = norm_col(cand)
            if key in by_norm:
                return by_norm[key]

        tokens_list = [[t for t in cand.lower().split()] for cand in cands]
        for n, orig in by_norm.items():
            for toks in tokens_list:
                if all(t in n for t in toks):
                    return orig

        raise KeyError(f"Δεν βρέθηκε στήλη για {cands}")

    if _THEFT_DF is None:
        df = pd.read_excel(POSTCODES_XLSX, sheet_name=0, dtype=str)
        df.columns = [c.strip() for c in df.columns]

        col_two = _pick_col(["Two first PostCode", "Two fisrt PostCode", "Two first Postcode"], df.columns)
        col_t1 = _pick_col(["Theft"], df.columns)

        col_t2 = None
        try:
            col_t2 = _pick_col(["Theft2", "Theft_2"], df.columns)
        except Exception:
            pass

        df["_two"] = df[col_two].astype(str).str.replace(r"\D", "", regex=True).str[:2].str.lstrip("0")
        df[col_t1] = df[col_t1].astype(str).str.replace(",", ".").astype(float)

        if col_t2:
            df[col_t2] = df[col_t2].astype(str).str.replace(",", ".").astype(float)

        _THEFT_DF = df
        _THEFT_COLS = {"two": "_two", "Theft": col_t1, "Theft2": col_t2}

    two = _digits(postcode)[:2].lstrip("0")
    if not two:
        return None

    col = _THEFT_COLS.get(which)
    if not col:
        raise ValueError("Δώσε which='Theft' ή which='Theft2'")

    hit = _THEFT_DF[_THEFT_DF[_THEFT_COLS["two"]] == two]
    if hit.empty:
        return None

    return float(hit.iloc[0][col])


def theft_premium(postcode, asset_value):
    r = theft_rate(postcode, which="Theft")
    if r is None:
        r = 0.0
    premium = float(THEFT_DAMAGE_RATIO) * float(r) * float(asset_value)
    rate = premium / float(asset_value) if asset_value else 0.0
    return float(rate), float(premium)


# ==========================================
# 13. MAIN
# ==========================================
def run_all(
    lon,
    lat,
    asset_value,
    year_construction,
    floors_house,
    stability_number,
    material_house,
    post_code,
):
    t_all = Timer()

    eq_rate, eq_prem = earthquake_premium(
        lon=lon,
        lat=lat,
        asset_value=asset_value,
        year_construction=year_construction,
        floors_house=floors_house,
        stability_number=stability_number,
        material_house=material_house,
    )
    fi_rate, fi_prem = fire_premium(lon, lat, post_code, asset_value)
    fl_rate, fl_prem = flood_premium(lon, lat, asset_value)
    wi_rate, wi_prem = wind_premium(lon, lat, asset_value)
    sn_rate, sn_prem = snow_premium(lon, lat, asset_value)
    th_rate, th_prem = theft_premium(post_code, asset_value)

    total = eq_prem + fi_prem + fl_prem + wi_prem + sn_prem + th_prem
    total_sec = t_all.split()

    return {
        "earthquake": {"rate": eq_rate, "premium": eq_prem},
        "fire": {"rate": fi_rate, "premium": fi_prem},
        "flood": {"rate": fl_rate, "premium": fl_prem},
        "wind": {"rate": wi_rate, "premium": wi_prem},
        "snow": {"rate": sn_rate, "premium": sn_prem},
        "theft": {"rate": th_rate, "premium": th_prem},
        "total": {"premium": total, "seconds": total_sec}
    }
