import argparse
import os

import geopandas as gpd
import numpy as np
from shapely.ops import linemerge, unary_union

# Match NYC: segmentize every 50 ft, then later buffer 25 ft. Paris is in metres.
FEET_TO_METERS = 0.3048
SEGMENTIZE_DISTANCE_M = 50.0 * FEET_TO_METERS
PROJ = "EPSG:2154"

LABEL_COLS = [
    "sidewalk_id",
    "num_pave",
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
]


def merge_centerline_pieces(geoms):
    """Merge one sidewalk's 2-point width segments back into a centerline."""
    unioned = unary_union(list(geoms))
    if unioned is None or unioned.is_empty:
        return None
    if unioned.geom_type == "LineString":
        return unioned
    if unioned.geom_type == "MultiLineString":
        return linemerge(unioned)
    if unioned.geom_type == "GeometryCollection":
        lines = [geom for geom in unioned.geoms if geom.geom_type in ("LineString", "MultiLineString")]
        if not lines:
            return None
        return linemerge(unary_union(lines))
    return None


def rebuild_centerlines(widths):
    """Dissolve short width segments into one centerline per sidewalk_id."""
    rows = []
    grouped = widths.groupby("sidewalk_id", dropna=False)
    for sidewalk_id, group in grouped:
        geom = merge_centerline_pieces(group.geometry)
        if geom is None or geom.is_empty:
            continue
        first = group.iloc[0]
        row = {col: first[col] if col in group.columns else None for col in LABEL_COLS}
        row["sidewalk_id"] = sidewalk_id
        if "width_m" in group.columns:
            row["width_m"] = float(group["width_m"].mean())
        row["geometry"] = geom
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=widths.crs)


def main(args):
    """Sample sidewalk centerlines every 50 ft, matching the NYC segmentize step."""
    root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(root, "data", "processed")
    widths_path = os.path.join(processed_dir, "sidewalk_widths_paris.geojson")
    out_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")

    if os.path.exists(out_path) and not args.force:
        print(f"Using existing file: {out_path}")
        return

    print(f"Loading sidewalk width segments from {widths_path} ...")
    widths = gpd.read_file(widths_path).to_crs(PROJ)
    widths = widths[widths.geometry.notna()].copy()
    widths = widths[~widths.geometry.is_empty].copy()
    print(f"Loaded {len(widths)} width segments")

    print("Merging width segments into sidewalk centerlines ...")
    centerlines = rebuild_centerlines(widths)
    centerlines = centerlines.reset_index(drop=True)
    print(f"Rebuilt {len(centerlines)} sidewalk centerlines")

    print(f"Segmentizing every {SEGMENTIZE_DISTANCE_M:.3f} m (50 ft) ...")
    sampled = (
        centerlines.geometry.segmentize(SEGMENTIZE_DISTANCE_M)
        .extract_unique_points()
        .explode(index_parts=True)
    )
    points = gpd.GeoDataFrame(geometry=sampled, crs=PROJ).reset_index()
    points = points.merge(
        centerlines.drop(columns=["geometry"]),
        left_on="level_0",
        right_index=True,
        how="left",
    )
    points = points.drop(columns=["level_0", "level_1"])
    points["segment_index"] = points["sidewalk_id"]
    points["point_index"] = np.arange(len(points), dtype=int)
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=PROJ)

    print(f"Sample points: {len(points)}")
    print(f"Writing {out_path} ...")
    points.to_file(out_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentize Paris sidewalk centerlines into NYC-style sample points."
    )
    parser.add_argument("--force", action="store_true", help="Regenerate the segmentized points even if the file exists")
    args = parser.parse_args()
    main(args)
