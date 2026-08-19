import argparse
import json
import os
import sys
from urllib.parse import quote
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import PROJ, normalize_with_quantile_clamping

FEET_TO_METERS = 0.3048
MAX_NEAREST_DISTANCE_M = 50.0 * FEET_TO_METERS
TRAFFIC_DAY = "2026-06-01"


def fetch_daily_occupation(date_string):
    """Query the Paris traffic API for one day and return mean occupation rate by iu_ac."""
    where = f"t_1h >= date'{date_string}' AND t_1h < date'{date_string[:8]}02'"
    if date_string != "2026-06-01":
        end_day = pd.Timestamp(date_string) + pd.Timedelta(days=1)
        where = f"t_1h >= date'{date_string}' AND t_1h < date'{end_day.strftime('%Y-%m-%d')}'"
    url = (
        "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
        "comptages-routiers-permanents/records?"
        f"select={quote('iu_ac,avg(k) as vehicle_traffic_occupation_raw')}"
        f"&where={quote(where)}"
        "&group_by=iu_ac&limit=10000"
    )
    with urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    traffic = pd.DataFrame(payload["results"])
    traffic["iu_ac"] = traffic["iu_ac"].astype(str)
    traffic["vehicle_traffic_occupation_raw"] = pd.to_numeric(
        traffic["vehicle_traffic_occupation_raw"], errors="coerce"
    ).fillna(0.0)
    return traffic


def main(args):
    """Assign mean occupation rate from nearest traffic-count arc within 50 ft, else 0."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    referential_path = os.path.join(raw_dir, "referentiel_comptages_routiers_raw.geojson")
    output_path = os.path.join(processed_dir, "vehicle_traffic_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(referential_path):
        raise FileNotFoundError(f"Missing {referential_path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Querying average occupation rate for {args.traffic_day} ...")
    traffic = fetch_daily_occupation(args.traffic_day)
    print(f"Loaded {len(traffic)} traffic arcs with daily occupation rates")

    print("Loading traffic referential ...")
    referential = gpd.read_file(referential_path).to_crs(PROJ)
    referential["iu_ac"] = referential["iu_ac"].astype(str)
    referential = referential.merge(traffic, on="iu_ac", how="left")
    referential["vehicle_traffic_occupation_raw"] = pd.to_numeric(
        referential["vehicle_traffic_occupation_raw"], errors="coerce"
    ).fillna(0.0)
    print(f"Loaded {len(referential)} referential arcs")

    print(f"Assigning nearest traffic arc within {args.max_nearest_distance_m:.2f} m (50 ft) ...")
    joined = gpd.sjoin_nearest(
        points[["point_index", "geometry"]],
        referential[["iu_ac", "vehicle_traffic_occupation_raw", "geometry"]],
        how="left",
        max_distance=args.max_nearest_distance_m,
        distance_col="vehicle_traffic_distance_m",
    )
    joined = joined.sort_values("point_index").drop_duplicates("point_index")
    points = points.merge(
        joined[["point_index", "iu_ac", "vehicle_traffic_occupation_raw", "vehicle_traffic_distance_m"]],
        on="point_index",
        how="left",
    )
    points["vehicle_traffic_occupation_raw"] = points["vehicle_traffic_occupation_raw"].fillna(0.0)
    points["vehicle_traffic_distance_m"] = points["vehicle_traffic_distance_m"].fillna(args.max_nearest_distance_m)

    points["vehicle_traffic_score"], lo, hi = normalize_with_quantile_clamping(
        points["vehicle_traffic_occupation_raw"]
    )
    print(f"Occupation rate quantiles: {lo:.3f} -> {hi:.3f}")

    keep_columns = [
        "point_index",
        "segment_index",
        "sidewalk_id",
        "pvp_tile",
        "arrondissement_id",
        "arrondissement",
        "qa_n_sq_qu",
        "qa_c_qu",
        "qa_c_quinsee",
        "qa_l_qu",
        "qa_c_ar",
        "qa_n_sq_ar",
        "iu_ac",
        "vehicle_traffic_distance_m",
        "vehicle_traffic_occupation_raw",
        "vehicle_traffic_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"vehicle_traffic_score stats:\n{points['vehicle_traffic_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Paris vehicle-traffic score from daily occupation rates.")
    parser.add_argument(
        "--traffic_day",
        default=TRAFFIC_DAY,
        help="Day to query in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--max_nearest_distance_m",
        type=float,
        default=MAX_NEAREST_DISTANCE_M,
        help="Maximum nearest-road distance in metres (50 ft default)",
    )
    args = parser.parse_args()
    main(args)
