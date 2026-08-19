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

BASE_SCORE = 0.3
AIRE_PIETONNE_SCORE = 0.7
ZONE_RENCONTRE_SCORE = 0.7
ZTL_BONUS = 0.1
PARIS_RESPIRE_BONUS = 0.2


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
    """Score zoning regulation: 0.3 base, 0.7 in aire piétonne/zone de rencontre, +0.1 ZTL, +0.2 Paris Respire."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    aires_path = os.path.join(raw_dir, "aires_pietonnes_raw.geojson")
    zones_path = os.path.join(raw_dir, "zones_de_rencontre_raw.geojson")
    ztl_path = os.path.join(raw_dir, "ztl_raw.geojson")
    respire_path = os.path.join(raw_dir, "paris_respire_raw.geojson")
    output_path = os.path.join(processed_dir, "zoning_regulation_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    for path in [aires_path, zones_path, ztl_path, respire_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading aires piétonnes ...")
    aires = flatten_z_geometries(load_features(aires_path))
    print(f"Loaded {len(aires)} aires piétonnes")

    print("Loading zones de rencontre ...")
    zones = flatten_z_geometries(load_features(zones_path))
    print(f"Loaded {len(zones)} zones de rencontre")

    print("Loading ZTL ...")
    ztl = flatten_z_geometries(load_features(ztl_path))
    print(f"Loaded {len(ztl)} ZTL polygons")

    print("Loading Paris Respire secteurs ...")
    respire = flatten_z_geometries(load_features(respire_path))
    print(f"Loaded {len(respire)} Paris Respire secteurs")

    print("Flagging points in aires piétonnes ...")
    points["zoning_regulation_aire_pietonne_flag"] = flag_points_in_polygons(
        points, aires, "zoning_regulation_aire_pietonne_flag"
    ).values

    print("Flagging points in zones de rencontre ...")
    points["zoning_regulation_zone_rencontre_flag"] = flag_points_in_polygons(
        points, zones, "zoning_regulation_zone_rencontre_flag"
    ).values

    print("Flagging points in ZTL ...")
    points["zoning_regulation_ztl_flag"] = flag_points_in_polygons(
        points, ztl, "zoning_regulation_ztl_flag"
    ).values

    print("Flagging points in Paris Respire secteurs ...")
    points["zoning_regulation_paris_respire_flag"] = flag_points_in_polygons(
        points, respire, "zoning_regulation_paris_respire_flag"
    ).values

    for col in [
        "zoning_regulation_aire_pietonne_flag",
        "zoning_regulation_zone_rencontre_flag",
        "zoning_regulation_ztl_flag",
        "zoning_regulation_paris_respire_flag",
    ]:
        print(f"  {col}: {(points[col] > 0).sum()} / {len(points)} points")

    # Start at base, override to calmed-zone score, then add bonuses.
    points["zoning_regulation_score"] = BASE_SCORE
    calmed = (points["zoning_regulation_aire_pietonne_flag"] > 0) | (
        points["zoning_regulation_zone_rencontre_flag"] > 0
    )
    points.loc[calmed, "zoning_regulation_score"] = ZONE_RENCONTRE_SCORE
    points["zoning_regulation_score"] += (
        ZTL_BONUS * points["zoning_regulation_ztl_flag"].fillna(0.0)
        + PARIS_RESPIRE_BONUS * points["zoning_regulation_paris_respire_flag"].fillna(0.0)
    )
    points["zoning_regulation_score"] = points["zoning_regulation_score"].clip(0.0, 1.0)

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
        "zoning_regulation_aire_pietonne_flag",
        "zoning_regulation_zone_rencontre_flag",
        "zoning_regulation_ztl_flag",
        "zoning_regulation_paris_respire_flag",
        "zoning_regulation_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(
        "zoning_regulation_score stats:\n"
        f"{points['zoning_regulation_score'].describe().to_string()}"
    )
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris zoning-regulation score from pedestrian zones, ZTL, and Paris Respire sectors."
    )
    args = parser.parse_args()
    main(args)
