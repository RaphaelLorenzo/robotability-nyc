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
    NEAR_BUFFER_DISTANCE_M,
    PROJ,
    aggregate_features_in_buffers,
    buffer_sample_points,
    load_features,
    normalize_with_quantile_clamping,
)

BASE_SCORE = 0.75
ZONE_RENCONTRE_SCORE = 0.9
AIRE_PIETONNE_SCORE = 1.0
ACCIDENT_MALUS_WEIGHT = 0.2
ANOMALY_MALUS_WEIGHT = 0.25


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


def normalize_minmax(series):
    """Scale a numeric series to 0-1 using min and max, without quantile clamping."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lower_value = values.min()
    upper_value = values.max()
    if pd.isna(lower_value) or pd.isna(upper_value) or upper_value <= lower_value:
        return pd.Series(0.0, index=series.index, dtype=float), float(lower_value), float(upper_value)
    normalized = (values - lower_value) / (upper_value - lower_value)
    return normalized.astype(float), float(lower_value), float(upper_value)


def load_tagged_soustypes(csv_path, variable_name):
    """Return Dans Ma Rue soustypes tagged with the requested variable."""
    soustypes = pd.read_csv(csv_path)
    tagged = soustypes[soustypes["variable"].astype(str).str.strip() == variable_name].copy()
    tagged["soustype"] = tagged["soustype"].astype(str).str.strip()
    tagged = tagged[tagged["soustype"].notna() & (tagged["soustype"] != "") & (tagged["soustype"] != "nan")]
    print(f"{variable_name} Dans Ma Rue soustypes ({len(tagged)}):")
    for row in tagged.itertuples(index=False):
        print(f"  - {row.type} / {row.soustype}")
    return tagged["soustype"].tolist()


def flag_points_in_polygons(points, polygons, column_name):
    """Return a 0/1 series for sample points that intersect any polygon."""
    flag = pd.Series(0.0, index=points["point_index"], name=column_name)
    if polygons.empty:
        return flag
    joined = gpd.sjoin(
        points[["point_index", "geometry"]],
        polygons[["geometry"]],
        how="left",
        predicate="intersects",
    )
    in_zone = joined.loc[joined["index_right"].notna(), "point_index"].unique()
    flag.loc[flag.index.isin(in_zone)] = 1.0
    return flag


def insee_arm_to_c_ar(code_series):
    """Map INSEE arrondissement codes such as 75116 to c_ar 1-20."""
    codes = pd.to_numeric(code_series, errors="coerce")
    return (codes % 100).astype("Int64")


def load_accident_victims(path):
    """Load the accidentologie victim table, which is exported as semicolon CSV."""
    usecols = ["com_arm_code", "id_accident", "victime_type"]
    accidents = pd.read_csv(path, sep=";", low_memory=False, usecols=lambda col: col in usecols)
    if "com_arm_code" not in accidents.columns:
        accidents = pd.read_csv(path, low_memory=False, usecols=lambda col: col in usecols)
    return accidents


def accident_rate_by_arrondissement(accidents_path, arrondissements_path):
    """Count unique accidents per km2 in each arrondissement and min-max scale to 0-1."""
    arrondissements = load_features(arrondissements_path)
    arrondissements = arrondissements[["c_ar", "l_ar", "geometry"]].copy()
    arrondissements["arrondissement_id"] = pd.to_numeric(arrondissements["c_ar"], errors="coerce").astype("Int64")
    arrondissements["arrondissement_area_km2"] = arrondissements.geometry.area / 1_000_000.0

    accidents = load_accident_victims(accidents_path)
    accidents["arrondissement_id"] = insee_arm_to_c_ar(accidents["com_arm_code"])
    accidents = accidents[accidents["arrondissement_id"].notna()].copy()
    print(f"Accident victim rows with an arrondissement: {len(accidents)}")
    if "victime_type" in accidents.columns:
        print("Victim mode counts:\n" + accidents["victime_type"].value_counts(dropna=False).to_string())

    if "id_accident" in accidents.columns:
        accident_counts = (
            accidents.dropna(subset=["id_accident"])
            .drop_duplicates(subset=["id_accident", "arrondissement_id"])
            .groupby("arrondissement_id")
            .size()
            .astype(float)
            .rename("intersection_safety_accident_count_raw")
        )
    else:
        accident_counts = (
            accidents.groupby("arrondissement_id").size().astype(float).rename("intersection_safety_accident_count_raw")
        )

    arrondissements = arrondissements.merge(accident_counts.reset_index(), on="arrondissement_id", how="left")
    arrondissements["intersection_safety_accident_count_raw"] = arrondissements[
        "intersection_safety_accident_count_raw"
    ].fillna(0.0)
    area_km2 = arrondissements["arrondissement_area_km2"].where(arrondissements["arrondissement_area_km2"] > 0)
    arrondissements["intersection_safety_accident_rate_raw"] = (
        arrondissements["intersection_safety_accident_count_raw"] / area_km2
    ).fillna(0.0)
    arrondissements["intersection_safety_accident_rate_score"], lower_value, upper_value = normalize_minmax(
        arrondissements["intersection_safety_accident_rate_raw"]
    )
    print(f"Accident rate per km2 min-max: {lower_value:.3f} -> {upper_value:.3f}")
    print(
        arrondissements[
            [
                "arrondissement_id",
                "l_ar",
                "arrondissement_area_km2",
                "intersection_safety_accident_count_raw",
                "intersection_safety_accident_rate_raw",
                "intersection_safety_accident_rate_score",
            ]
        ]
        .sort_values("arrondissement_id")
        .to_string(index=False)
    )
    return arrondissements


def main(args):
    """Score intersection safety from calmed zones, arrondissement accident rates, and Dans Ma Rue anomalies."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    aires_path = os.path.join(raw_dir, "aires_pietonnes_raw.geojson")
    zones_path = os.path.join(raw_dir, "zones_de_rencontre_raw.geojson")
    accidents_path = os.path.join(raw_dir, "accidentologie_victimes.csv")
    arrondissements_path = os.path.join(raw_dir, "arrondissements_raw.geojson")
    dans_ma_rue_path = os.path.join(raw_dir, "dans_ma_rue_raw.geojson")
    soustypes_path = os.path.join(root, "dans_ma_rue_soustypes.csv")
    output_path = os.path.join(processed_dir, "intersection_safety_paris.geojson")
    arrondissement_output_path = os.path.join(processed_dir, "intersection_safety_arrondissements.geojson")

    missing = [path for path in [aires_path, zones_path, accidents_path] if not os.path.exists(path)]
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run paris_test/segmentize_sidewalks.py first.")
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run paris_test/download_data.py first.")
    if not os.path.exists(dans_ma_rue_path):
        raise FileNotFoundError(f"Missing {dans_ma_rue_path}. Run paris_test/download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    points["arrondissement_id"] = pd.to_numeric(points["arrondissement_id"], errors="coerce").astype("Int64")
    print(f"Loaded {len(points)} sample points")

    print("Flagging points inside aires pietonnes and zones de rencontre ...")
    aires = flatten_z_geometries(load_features(aires_path))
    zones = flatten_z_geometries(load_features(zones_path))
    points["intersection_safety_aire_pietonne_flag"] = flag_points_in_polygons(
        points, aires, "intersection_safety_aire_pietonne_flag"
    ).to_numpy()
    points["intersection_safety_zone_rencontre_flag"] = flag_points_in_polygons(
        points, zones, "intersection_safety_zone_rencontre_flag"
    ).to_numpy()
    print(
        "Sample points in aire pietonne: "
        f"{int((points['intersection_safety_aire_pietonne_flag'] > 0).sum())} / {len(points)}"
    )
    print(
        "Sample points in zone de rencontre: "
        f"{int((points['intersection_safety_zone_rencontre_flag'] > 0).sum())} / {len(points)}"
    )

    points["intersection_safety_zone_score"] = BASE_SCORE
    points.loc[points["intersection_safety_zone_rencontre_flag"] > 0, "intersection_safety_zone_score"] = (
        ZONE_RENCONTRE_SCORE
    )
    points.loc[points["intersection_safety_aire_pietonne_flag"] > 0, "intersection_safety_zone_score"] = (
        AIRE_PIETONNE_SCORE
    )

    print("Computing unique accidents per km2 by arrondissement ...")
    arrondissements = accident_rate_by_arrondissement(accidents_path, arrondissements_path)
    accident_cols = [
        "arrondissement_id",
        "arrondissement_area_km2",
        "intersection_safety_accident_count_raw",
        "intersection_safety_accident_rate_raw",
        "intersection_safety_accident_rate_score",
    ]
    points = points.drop(columns=[col for col in accident_cols if col != "arrondissement_id"], errors="ignore")
    points = points.merge(arrondissements[accident_cols], on="arrondissement_id", how="left")
    for col in accident_cols[1:]:
        points[col] = points[col].fillna(0.0)

    print(f"Buffering sample points by {args.near_buffer_distance_m:.3f} m (25 ft) ...")
    near_buffers = buffer_sample_points(points, args.near_buffer_distance_m)

    print("Loading Dans Ma Rue reports tagged as intersection_safety ...")
    tagged_soustypes = load_tagged_soustypes(soustypes_path, "intersection_safety")
    anomalies = gpd.read_file(dans_ma_rue_path, columns=["soustype", "geometry"]).to_crs(PROJ)
    anomalies = anomalies[anomalies.geometry.notna() & ~anomalies.geometry.is_empty].copy()
    anomalies = flatten_z_geometries(anomalies)
    anomalies["soustype"] = anomalies["soustype"].astype(str).str.strip()
    anomalies = anomalies[anomalies["soustype"].isin(tagged_soustypes)].copy()
    print(f"Dans Ma Rue intersection-safety reports: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies["soustype"].value_counts().to_string())

    points["intersection_safety_anomaly_count_raw"] = aggregate_features_in_buffers(
        near_buffers,
        anomalies,
        output_column="intersection_safety_anomaly_count_raw",
    ).to_numpy()
    points["intersection_safety_anomaly_count_score"], lower_value, upper_value = normalize_with_quantile_clamping(
        points["intersection_safety_anomaly_count_raw"]
    )
    print(f"Anomaly count quantiles: {lower_value:.3f} -> {upper_value:.3f}")

    points["intersection_safety_score"] = (
        points["intersection_safety_zone_score"].fillna(BASE_SCORE)
        - ACCIDENT_MALUS_WEIGHT * points["intersection_safety_accident_rate_score"].fillna(0.0)
        - ANOMALY_MALUS_WEIGHT * points["intersection_safety_anomaly_count_score"].fillna(0.0)
    )
    points["intersection_safety_score"] = points["intersection_safety_score"].clip(0.0, 1.0)

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
        "intersection_safety_aire_pietonne_flag",
        "intersection_safety_zone_rencontre_flag",
        "intersection_safety_zone_score",
        "arrondissement_area_km2",
        "intersection_safety_accident_count_raw",
        "intersection_safety_accident_rate_raw",
        "intersection_safety_accident_rate_score",
        "intersection_safety_anomaly_count_raw",
        "intersection_safety_anomaly_count_score",
        "intersection_safety_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"intersection_safety_score stats:\n{points['intersection_safety_score'].describe().to_string()}")
    print(f"Writing {arrondissement_output_path} ...")
    arrondissements.to_file(arrondissement_output_path, driver="GeoJSON")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris intersection safety from calmed zones, accident rates, and Dans Ma Rue anomalies."
    )
    parser.add_argument(
        "--near_buffer_distance_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Buffer radius for Dans Ma Rue anomalies, in metres (25 ft)",
    )
    args = parser.parse_args()
    main(args)
