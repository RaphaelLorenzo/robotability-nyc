import argparse
import os
import sys

import geopandas as gpd
import shapely

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import (
    NEAR_BUFFER_DISTANCE_M,
    PROJ,
    aggregate_features_in_buffers,
    buffer_sample_points,
    load_features,
    normalize_with_quantile_clamping,
)


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


def main(args):
    """Count public lamps within 25 ft of each sidewalk point, normalize 2.5-99.5% to 0-1."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    eclairage_path = os.path.join(raw_dir, "eclairage_public_raw.geojson")
    output_path = os.path.join(processed_dir, "street_lighting_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(eclairage_path):
        raise FileNotFoundError(f"Missing {eclairage_path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Buffering sample points by {args.buffer_distance_m:.1f} m (25 ft) ...")
    near_buffers = buffer_sample_points(points, args.buffer_distance_m)

    print("Loading éclairage public (lamps) ...")
    lamps = flatten_z_geometries(load_features(eclairage_path))
    print(f"Loaded {len(lamps)} lamps")

    print("Counting lamps in 25 ft buffers ...")
    points["street_lighting_lamp_count_raw"] = aggregate_features_in_buffers(
        near_buffers, lamps, output_column="street_lighting_lamp_count_raw"
    ).to_numpy()

    points["street_lighting_score"], lo, hi = normalize_with_quantile_clamping(
        points["street_lighting_lamp_count_raw"]
    )
    print(f"Lamp count quantiles: {lo:.3f} -> {hi:.3f}")

    keep_columns = [
        "point_index", "segment_index", "sidewalk_id", "pvp_tile",
        "arrondissement_id", "arrondissement",
        "qa_n_sq_qu", "qa_c_qu", "qa_c_quinsee", "qa_l_qu", "qa_c_ar", "qa_n_sq_ar",
        "street_lighting_lamp_count_raw",
        "street_lighting_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"street_lighting_score stats:\n{points['street_lighting_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris street-lighting score from éclairage public lamp count."
    )
    parser.add_argument(
        "--buffer_distance_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Buffer radius for lamp counting (25 ft default)",
    )
    args = parser.parse_args()
    main(args)
