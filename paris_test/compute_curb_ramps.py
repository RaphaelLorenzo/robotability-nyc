import argparse
import os
import sys

import geopandas as gpd
import shapely

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import PROJ, load_features
from compute_intersection_safety import flag_points_in_polygons

BASE_SCORE = 0.75
VOIE_ESCALIER_SCORE = 0.0
QUARTIER_ACCESSIBILITE_SCORE = 0.9


def flatten_z_geometries(gdf):
    """Drop Z coordinates so spatial joins stay 2D."""
    if gdf.geometry.has_z.any():
        gdf = gdf.copy()
        gdf["geometry"] = gpd.GeoSeries(
            shapely.force_2d(gdf.geometry.values),
            index=gdf.index,
            crs=gdf.crs,
        )
    return gdf


def buffer_line_features(gdf, buffer_m):
    """Buffer line/point features so that point-in-polygon logic picks them up."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    return gdf


def main(args):
    """Score curb-ramp availability: 0.75 base, 0 on stair streets, 0.9 in accessibility quarters."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    escalier_path = os.path.join(raw_dir, "voies_en_escalier_raw.geojson")
    accessibilite_path = os.path.join(raw_dir, "quartiers_accessibilite_raw.geojson")
    output_path = os.path.join(processed_dir, "curb_ramps_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    for path in [escalier_path, accessibilite_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading voies en escalier ...")
    escaliers = flatten_z_geometries(load_features(escalier_path))
    print(f"Loaded {len(escaliers)} stair streets")
    # Buffer lines by 5 m so sample points on adjacent sidewalks are caught.
    escaliers_buffered = buffer_line_features(escaliers, args.escalier_buffer_m)

    print("Loading quartiers d'accessibilité augmentée ...")
    accessibilite = flatten_z_geometries(load_features(accessibilite_path))
    print(f"Loaded {len(accessibilite)} accessibility quarters")

    print("Flagging points on stair streets ...")
    points["curb_ramps_escalier_flag"] = flag_points_in_polygons(
        points, escaliers_buffered, "curb_ramps_escalier_flag"
    ).values

    print("Flagging points in accessibility quarters ...")
    points["curb_ramps_accessibilite_flag"] = flag_points_in_polygons(
        points, accessibilite, "curb_ramps_accessibilite_flag"
    ).values

    escalier_count = (points["curb_ramps_escalier_flag"] > 0).sum()
    accessibilite_count = (points["curb_ramps_accessibilite_flag"] > 0).sum()
    print(f"Points on stair streets: {escalier_count} / {len(points)}")
    print(f"Points in accessibility quarters: {accessibilite_count} / {len(points)}")

    # Apply scoring: start at base, override with accessibility quarter, then
    # override with stair street (lowest priority, hardest constraint).
    points["curb_ramps_score"] = BASE_SCORE
    points.loc[points["curb_ramps_accessibilite_flag"] > 0, "curb_ramps_score"] = QUARTIER_ACCESSIBILITE_SCORE
    points.loc[points["curb_ramps_escalier_flag"] > 0, "curb_ramps_score"] = VOIE_ESCALIER_SCORE

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
        "curb_ramps_escalier_flag",
        "curb_ramps_accessibilite_flag",
        "curb_ramps_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(
        "curb_ramps_score stats:\n"
        f"{points['curb_ramps_score'].describe().to_string()}"
    )
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris curb-ramp availability score from stair streets and accessibility quarters."
    )
    parser.add_argument(
        "--escalier_buffer_m",
        type=float,
        default=5.0,
        help="Buffer around stair-street lines to catch nearby sample points (metres)",
    )
    args = parser.parse_args()
    main(args)
