import argparse
import os
import sys

import geopandas as gpd
import pandas as pd
import shapely

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import (
    PROJ,
    load_features,
    normalize_qa_code_paris,
    normalize_with_quantile_clamping,
)

ZTI_SCORE = 0.5
TOURIST_SITES_WEIGHT = 0.5


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


def count_tourist_sites_per_qa(qa, tourist_sites_path):
    """Count tourist sites inside each quartier administratif and scale to 0-1."""
    tourist_sites = flatten_z_geometries(load_features(tourist_sites_path))
    tourist_joined = gpd.sjoin(
        tourist_sites,
        qa[["qa_c_quinsee", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    tourist_counts = (
        tourist_joined.groupby("qa_c_quinsee")
        .size()
        .astype(float)
        .rename("crowd_dynamics_tourist_sites_count_raw")
    )
    qa = qa.merge(tourist_counts.reset_index(), on="qa_c_quinsee", how="left")
    qa["crowd_dynamics_tourist_sites_count_raw"] = qa["crowd_dynamics_tourist_sites_count_raw"].fillna(0.0)
    qa["crowd_dynamics_tourist_sites_count_score"], lower_value, upper_value = normalize_with_quantile_clamping(
        qa["crowd_dynamics_tourist_sites_count_raw"]
    )
    print(f"Tourist site counts in Paris QAs: {int(qa['crowd_dynamics_tourist_sites_count_raw'].sum())}")
    print(f"Tourist site count quantiles: {lower_value:.3f} -> {upper_value:.3f}")
    return qa


def attach_qa_values(points, qa, value_cols):
    """Join QA attributes onto sample points by location."""
    qa_values = qa[["qa_c_quinsee"] + value_cols + ["geometry"]].copy()
    points = points.drop(columns=value_cols, errors="ignore")
    qa_joined = gpd.sjoin(
        points[["point_index", "geometry"]],
        qa_values,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    qa_joined = qa_joined.drop_duplicates(subset=["point_index"], keep="first")
    missing = qa_joined[value_cols[0]].isna()
    if missing.any():
        nearest_qa = gpd.sjoin_nearest(
            points.loc[points["point_index"].isin(qa_joined.loc[missing, "point_index"]), ["point_index", "geometry"]],
            qa_values,
            how="left",
            max_distance=50,
        ).drop(columns=["index_right"], errors="ignore")
        nearest_qa = nearest_qa.drop_duplicates(subset=["point_index"], keep="first")
        qa_joined = pd.concat([qa_joined.loc[~missing], nearest_qa], ignore_index=True)
    points = points.merge(qa_joined.drop(columns=["geometry", "qa_c_quinsee"], errors="ignore"), on="point_index", how="left")
    points[value_cols] = points[value_cols].fillna(0.0)
    return points


def flag_points_in_zti(points, zti_path):
    """Assign 0.5 to every sample point that falls inside a ZTI polygon."""
    zti = flatten_z_geometries(load_features(zti_path))
    zti = zti[["geometry"]].copy()
    joined = gpd.sjoin(points[["point_index", "geometry"]], zti, how="left", predicate="intersects")
    in_zti = joined.loc[joined["index_right"].notna(), "point_index"].unique()
    points["crowd_dynamics_zti_score"] = 0.0
    points.loc[points["point_index"].isin(in_zti), "crowd_dynamics_zti_score"] = ZTI_SCORE
    print(f"Sample points inside ZTI: {(points['crowd_dynamics_zti_score'] > 0).sum()} / {len(points)}")
    return points


def main(args):
    """Score crowd dynamics from ZTI membership and QA tourist-site counts, then invert for NYC polarity."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    qa_path = os.path.join(raw_dir, "quartier_paris_raw.geojson")
    tourist_sites_path = os.path.join(raw_dir, "sites_touristiques_raw.geojson")
    zti_path = os.path.join(raw_dir, "zones_touristiques_internationales_raw.geojson")
    output_path = os.path.join(processed_dir, "crowd_dynamics_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run paris_test/segmentize_sidewalks.py first.")
    if not os.path.exists(zti_path):
        raise FileNotFoundError(f"Missing {zti_path}. Run paris_test/download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading quartier administratif polygons ...")
    qa = gpd.read_file(qa_path).to_crs(PROJ)
    qa["qa_c_quinsee"] = normalize_qa_code_paris(qa["c_quinsee"], qa["c_qu"])

    print("Counting tourist sites per QA ...")
    qa = count_tourist_sites_per_qa(qa, tourist_sites_path)
    tourist_cols = ["crowd_dynamics_tourist_sites_count_raw", "crowd_dynamics_tourist_sites_count_score"]
    points = attach_qa_values(points, qa, tourist_cols)

    print("Flagging points inside Zones Touristiques Internationales ...")
    points = flag_points_in_zti(points, zti_path)

    points["crowd_dynamics_tourism_score"] = (
        points["crowd_dynamics_zti_score"].fillna(0.0)
        + TOURIST_SITES_WEIGHT * points["crowd_dynamics_tourist_sites_count_score"].fillna(0.0)
    )
    points["crowd_dynamics_tourism_score"] = points["crowd_dynamics_tourism_score"].clip(0.0, 1.0)
    # NYC polarity: residential/predictable purpose is high, tourist/commercial purpose is low.
    points["crowd_dynamics_score"] = 1.0 - points["crowd_dynamics_tourism_score"]

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
        "crowd_dynamics_zti_score",
        "crowd_dynamics_tourist_sites_count_raw",
        "crowd_dynamics_tourist_sites_count_score",
        "crowd_dynamics_tourism_score",
        "crowd_dynamics_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"crowd_dynamics_score stats:\n{points['crowd_dynamics_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris crowd dynamics from ZTI zones and QA tourist-site counts."
    )
    args = parser.parse_args()
    main(args)
