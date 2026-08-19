import os

import geopandas as gpd

from compute_pedestrian_density import PROJ


def compute_constant_feature(feature_name, feature_value):
    """Write a constant-valued feature GeoJSON over the segmentized sidewalk points."""
    root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    output_path = os.path.join(processed_dir, f"{feature_name}_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")

    points = gpd.read_file(points_path).to_crs(PROJ)
    points[f"{feature_name}_score"] = float(feature_value)

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
        f"{feature_name}_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()
    points.to_file(output_path, driver="GeoJSON")
    print(f"Wrote {output_path}")
