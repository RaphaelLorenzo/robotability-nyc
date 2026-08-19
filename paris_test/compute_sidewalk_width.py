import argparse
import os
import sys

import geopandas as gpd
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import PROJ, normalize_with_quantile_clamping


def main(args):
    """Normalize segmentized sidewalk width to a 0-1 feature with 2.5%-99.5% clamping."""
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    output_path = os.path.join(processed_dir, "sidewalk_width_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    points["sidewalk_width_raw"] = pd.to_numeric(points["width_m"], errors="coerce")
    points["sidewalk_width_score"], lo, hi = normalize_with_quantile_clamping(points["sidewalk_width_raw"])
    print(f"Sidewalk width quantiles: {lo:.3f} -> {hi:.3f}")

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
        "width_m",
        "sidewalk_width_raw",
        "sidewalk_width_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"sidewalk_width_score stats:\n{points['sidewalk_width_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Paris sidewalk-width score from segmentized width_m values.")
    args = parser.parse_args()
    main(args)
