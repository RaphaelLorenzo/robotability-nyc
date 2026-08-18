import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# Near features stay on the sidewalk (25 ft). Building-scale amenities use 200 ft.
FEET_TO_METERS = 0.3048
NEAR_BUFFER_DISTANCE_M = 25.0 * FEET_TO_METERS
AMENITY_BUFFER_DISTANCE_M = 200.0 * FEET_TO_METERS
PROJ = "EPSG:2154"

DEFAULT_UPPER_QUANTILE = 0.995
DEFAULT_LOWER_QUANTILE = 0.025


def normalize_with_quantile_clamping(series, lower_quantile=DEFAULT_LOWER_QUANTILE, upper_quantile=DEFAULT_UPPER_QUANTILE):
    """Clamp a numeric series to quantiles, then scale it to the 0-1 range."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lower_value = values.quantile(lower_quantile)
    upper_value = values.quantile(upper_quantile)

    if pd.isna(lower_value) or pd.isna(upper_value) or upper_value <= lower_value:
        return pd.Series(0.0, index=series.index, dtype=float), float(lower_value), float(upper_value)

    clipped = values.clip(lower=lower_value, upper=upper_value)
    normalized = (clipped - lower_value) / (upper_value - lower_value)
    return normalized.astype(float), float(lower_value), float(upper_value)


def normalize_qa_code_insee(qa_c_series):
    """Normalize QA INSEE codes so '7511003' and '7511003.0' match."""
    return pd.to_numeric(qa_c_series, errors="coerce").astype("Int64").astype(str)


def normalize_qa_code_paris(qa_c_series, c_qu_series):
    """Normalize paris's QA codes to match iris insee codes. For example 7511603 (Porte-Dauphine, with c_qu = 63) has IRIS code 7511663"""
    int_qa_c_codes = pd.to_numeric(qa_c_series, errors="coerce").astype("Int64")
    in_c_qu_series = pd.to_numeric(c_qu_series, errors="coerce").astype("Int64")
    # add a leading zero in_c_qu_series if length is less than 2
    in_c_qu_series = in_c_qu_series.astype(str).str.zfill(2)
    
    # keep only the leading 5 numbers of int_qa_c_codes and append the c_qu code
    normalized_qa_c_codes = int_qa_c_codes.astype(str).str.slice(0, 5) + in_c_qu_series.astype(str)
    return normalized_qa_c_codes


def load_population_by_qa(population_csv_path):
    """Aggregate 2022 population from IRIS rows to Paris quartier administratif codes."""
    population = pd.read_csv(population_csv_path, sep=";", low_memory=False)
    population = population[population["COM"].astype(str).str.startswith("751")].copy()

    population["qa_c_quinsee"] = normalize_qa_code_insee(population["IRIS"].astype(str).str.slice(0, 7))
    population["P22_POP"] = pd.to_numeric(population["P22_POP"], errors="coerce").fillna(0.0)
    
    ret = population.groupby("qa_c_quinsee", dropna=False)["P22_POP"].sum().reset_index().rename(columns={"P22_POP": "population_2022"})

    # Write the processed DataFrame to CSV for inspection/debugging
    dirname = os.path.dirname(population_csv_path)
    ret.to_csv(os.path.join(dirname, "processed_population_by_qa.csv"), index=False)
    print(f"Wrote {os.path.join(dirname, 'processed_population_by_qa.csv')}")

    return ret


def load_features(path):
    """Load a Paris amenity layer and project it to Lambert 93."""
    data = gpd.read_file(path).to_crs(PROJ)
    data = data[data.geometry.notna()].copy()
    data = data[~data.geometry.is_empty].copy()
    return data


def filter_activities_for_2025(activities):
    """Keep activities whose event date range overlaps calendar year 2025."""
    activities = activities.copy()
    activities["date_start"] = pd.to_datetime(activities["date_start"], utc=True, errors="coerce")
    activities["date_end"] = pd.to_datetime(activities["date_end"], utc=True, errors="coerce")

    start_2025 = pd.Timestamp("2025-01-01T00:00:00Z")
    end_2025 = pd.Timestamp("2025-12-31T23:59:59Z")

    has_date = activities["date_start"].notna() | activities["date_end"].notna()
    overlaps_2025 = (
        ((activities["date_start"].isna()) | (activities["date_start"] <= end_2025))
        & ((activities["date_end"].isna()) | (activities["date_end"] >= start_2025))
    )
    return activities[has_date & overlaps_2025].copy()


def aggregate_features_in_buffers(buffered_points, features, output_column, value_column=None):
    """Count or sum features that intersect each sample-point buffer, matching NYC sjoin."""
    if features.empty:
        return pd.Series(0.0, index=buffered_points["point_index"], name=output_column)

    joined = gpd.sjoin(
        buffered_points[["point_index", "geometry"]],
        features,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return pd.Series(0.0, index=buffered_points["point_index"], name=output_column)

    if value_column is None:
        aggregated = joined.groupby("point_index").size().astype(float).rename(output_column)
    else:
        aggregated = (
            pd.to_numeric(joined[value_column], errors="coerce")
            .fillna(0.0)
            .groupby(joined["point_index"])
            .sum()
            .rename(output_column)
        )
    return aggregated.reindex(buffered_points["point_index"]).fillna(0.0)


def buffer_sample_points(points, distance_m):
    """Return a point-index GeoDataFrame buffered at the requested radius."""
    buffered_points = points[["point_index", "geometry"]].copy()
    buffered_points["geometry"] = buffered_points.geometry.buffer(distance_m)
    return buffered_points


def main(args):
    """Compute pedestrian density on NYC-style buffered sidewalk sample points."""
    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, "data", "raw")
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    qa_path = os.path.join(raw_dir, "quartier_paris_raw.geojson")
    population_csv_path = os.path.join(raw_dir, "base-ic-evol-struct-pop-2022_csv", "base-ic-evol-struct-pop-2022.CSV")
    output_path = os.path.join(processed_dir, "pedestrian_density_paris.geojson")
    qa_output_path = os.path.join(processed_dir, "pedestrian_density_qa.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(
            f"Missing {points_path}. Run paris_test/segmentize_sidewalks.py first."
        )

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    points["qa_c_quinsee"] = normalize_qa_code_insee(points["qa_c_quinsee"])
    print(f"Loaded {len(points)} sample points")

    print("Aggregating 2022 population to QA ...")
    qa = gpd.read_file(qa_path).to_crs(PROJ)
    qa["qa_c_quinsee"] = normalize_qa_code_paris(qa["c_quinsee"], qa["c_qu"])
    population_by_qa = load_population_by_qa(population_csv_path)
    qa = qa.merge(population_by_qa, on="qa_c_quinsee", how="left")
    qa["population_2022"] = qa["population_2022"].fillna(0.0)
    print(qa.head())
    print(f"Got a total of {len(qa)} QA rows")
    
    
    qa["qa_surface_km2"] = qa.geometry.area / 1_000_000.0
    qa["pedestrian_density_base_density_raw"] = qa["population_2022"] / qa["qa_surface_km2"].replace(0, np.nan)
    qa["pedestrian_density_base_density_raw"] = (
        qa["pedestrian_density_base_density_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    qa["pedestrian_density_base_density_score"], base_lower, base_upper = normalize_with_quantile_clamping(
        qa["pedestrian_density_base_density_raw"]
    )
    print(f"Base density quantiles: {base_lower:.3f} -> {base_upper:.3f}")

    print("Counting tourist sites per QA ...")
    tourist_sites_path = os.path.join(raw_dir, "sites_touristiques_raw.geojson")
    tourist_sites = load_features(tourist_sites_path)
    if tourist_sites.geometry.has_z.any():
        tourist_sites["geometry"] = gpd.GeoSeries(shapely.force_2d(tourist_sites.geometry.values), index=tourist_sites.index, crs=tourist_sites.crs)
    tourist_joined = gpd.sjoin(
        tourist_sites,
        qa[["qa_c_quinsee", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    tourist_counts = (
        tourist_joined.groupby("qa_c_quinsee").size().astype(float).rename("pedestrian_density_tourist_sites_count_raw")
    )
    qa = qa.merge(tourist_counts.reset_index(), on="qa_c_quinsee", how="left")
    qa["pedestrian_density_tourist_sites_count_raw"] = qa["pedestrian_density_tourist_sites_count_raw"].fillna(0.0)
    qa["pedestrian_density_tourist_sites_count_score"], tourist_lower, tourist_upper = normalize_with_quantile_clamping(
        qa["pedestrian_density_tourist_sites_count_raw"]
    )
    print(f"Tourist site counts in Paris QAs: {int(qa['pedestrian_density_tourist_sites_count_raw'].sum())}")
    print(f"Tourist site count quantiles: {tourist_lower:.3f} -> {tourist_upper:.3f}")

    # Attach the QA density of the sample point location, not a whole-street average.
    qa_density = qa[
        [
            "qa_c_quinsee",
            "population_2022",
            "qa_surface_km2",
            "pedestrian_density_base_density_raw",
            "pedestrian_density_base_density_score",
            "pedestrian_density_tourist_sites_count_raw",
            "pedestrian_density_tourist_sites_count_score",
            "geometry",
        ]
    ].copy()
    density_cols = [
        "population_2022",
        "qa_surface_km2",
        "pedestrian_density_base_density_raw",
        "pedestrian_density_base_density_score",
        "pedestrian_density_tourist_sites_count_raw",
        "pedestrian_density_tourist_sites_count_score",
    ]
    points = points.drop(columns=density_cols, errors="ignore")
    qa_joined = gpd.sjoin(
        points[["point_index", "geometry"]],
        qa_density,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    qa_joined = qa_joined.drop_duplicates(subset=["point_index"], keep="first")
    missing_qa = qa_joined["pedestrian_density_base_density_raw"].isna()
    if missing_qa.any():
        nearest_qa = gpd.sjoin_nearest(
            points.loc[points["point_index"].isin(qa_joined.loc[missing_qa, "point_index"]), ["point_index", "geometry"]],
            qa_density,
            how="left",
            max_distance=50,
        ).drop(columns=["index_right"], errors="ignore")
        nearest_qa = nearest_qa.drop_duplicates(subset=["point_index"], keep="first")
        qa_joined = pd.concat([qa_joined.loc[~missing_qa], nearest_qa], ignore_index=True)
    points = points.merge(qa_joined.drop(columns=["geometry", "qa_c_quinsee"], errors="ignore"), on="point_index", how="left")
    n_with_density = points["pedestrian_density_base_density_raw"].notna().sum()
    print(f"Sample points with QA density: {n_with_density} / {len(points)}")
    points[density_cols] = points[density_cols].fillna(0.0)
    
    print(points.head())
    print(f"Got a total of {len(points)} points")

    print(f"Buffering sample points by {args.near_buffer_distance_m:.3f} m (25 ft) and {args.amenity_buffer_distance_m:.3f} m (200 ft) ...")
    near_buffers = buffer_sample_points(points, args.near_buffer_distance_m)
    amenity_buffers = buffer_sample_points(points, args.amenity_buffer_distance_m)

    amenity_count_sources = {
        "pedestrian_density_lieux_municipaux_count_raw": os.path.join(raw_dir, "lieux_municipaux_raw.geojson"),
        "pedestrian_density_colleges_count_raw": os.path.join(raw_dir, "colleges_raw.geojson"),
        "pedestrian_density_ecoles_elementaires_count_raw": os.path.join(raw_dir, "ecoles_elementaires_raw.geojson"),
        "pedestrian_density_ecoles_maternelles_count_raw": os.path.join(raw_dir, "ecoles_maternelles_raw.geojson"),
        "pedestrian_density_kiosques_de_presse_count_raw": os.path.join(raw_dir, "kiosques_de_presse_raw.geojson"),
    }
    near_count_sources = {
        "pedestrian_density_points_arrets_count_raw": os.path.join(raw_dir, "points_arrets_bus_raw.geojson"),
    }

    for column_name, path in amenity_count_sources.items():
        print(f"Counting {column_name} in 200 ft buffers ...")
        features = load_features(path)
        points[column_name] = aggregate_features_in_buffers(
            amenity_buffers,
            features,
            output_column=column_name,
        ).to_numpy()

    for column_name, path in near_count_sources.items():
        print(f"Counting {column_name} in 25 ft buffers ...")
        features = load_features(path)
        points[column_name] = aggregate_features_in_buffers(
            near_buffers,
            features,
            output_column=column_name,
        ).to_numpy()

    print("Summing terrasse and etalage surfaces in 25 ft buffers ...")
    terrasses = load_features(os.path.join(raw_dir, "terrasses_autorisations_raw.geojson"))
    terrasses["terrasse_surface_m2"] = (
        pd.to_numeric(terrasses["longueur"], errors="coerce").fillna(0.0)
        * pd.to_numeric(terrasses["largeur"], errors="coerce").fillna(0.0)
    )
    points["pedestrian_density_terrasses_surface_raw"] = aggregate_features_in_buffers(
        near_buffers,
        terrasses,
        output_column="pedestrian_density_terrasses_surface_raw",
        value_column="terrasse_surface_m2",
    ).to_numpy()

    print("Counting 2025 activities in 200 ft buffers ...")
    activities = filter_activities_for_2025(load_features(os.path.join(raw_dir, "activites_raw.geojson")))
    points["pedestrian_density_activities_2025_count_raw"] = aggregate_features_in_buffers(
        amenity_buffers,
        activities,
        output_column="pedestrian_density_activities_2025_count_raw",
    ).to_numpy()

    raw_component_columns = [
        "pedestrian_density_lieux_municipaux_count_raw",
        "pedestrian_density_colleges_count_raw",
        "pedestrian_density_ecoles_elementaires_count_raw",
        "pedestrian_density_ecoles_maternelles_count_raw",
        "pedestrian_density_kiosques_de_presse_count_raw",
        "pedestrian_density_points_arrets_count_raw",
        "pedestrian_density_terrasses_surface_raw",
        "pedestrian_density_activities_2025_count_raw",
    ]

    for raw_column in raw_component_columns:
        score_column = raw_column.replace("_raw", "_score")
        points[score_column], lower_value, upper_value = normalize_with_quantile_clamping(points[raw_column])
        print(f"{score_column} quantiles: {lower_value:.3f} -> {upper_value:.3f}")

    weights = {
        "pedestrian_density_base_density_score": 0.25,
        "pedestrian_density_tourist_sites_count_score": 0.25,
        "pedestrian_density_ecoles_elementaires_count_score": 0.05,
        "pedestrian_density_ecoles_maternelles_count_score": 0.05,
        "pedestrian_density_colleges_count_score": 0.05,
        "pedestrian_density_terrasses_surface_score": 0.1,
        "pedestrian_density_activities_2025_count_score": 0.1,
        "pedestrian_density_points_arrets_count_score": 0.1,
        "pedestrian_density_kiosques_de_presse_count_score": 0.05,
    }

    points["pedestrian_density_score"] = 0.0
    for column_name, weight in weights.items():
        points["pedestrian_density_score"] += points[column_name].fillna(0.0) * weight

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
        "population_2022",
        "qa_surface_km2",
        "pedestrian_density_base_density_raw",
        "pedestrian_density_base_density_score",
        "pedestrian_density_tourist_sites_count_raw",
        "pedestrian_density_tourist_sites_count_score",
        "pedestrian_density_lieux_municipaux_count_raw",
        "pedestrian_density_lieux_municipaux_count_score",
        "pedestrian_density_colleges_count_raw",
        "pedestrian_density_colleges_count_score",
        "pedestrian_density_ecoles_elementaires_count_raw",
        "pedestrian_density_ecoles_elementaires_count_score",
        "pedestrian_density_ecoles_maternelles_count_raw",
        "pedestrian_density_ecoles_maternelles_count_score",
        "pedestrian_density_kiosques_de_presse_count_raw",
        "pedestrian_density_kiosques_de_presse_count_score",
        "pedestrian_density_points_arrets_count_raw",
        "pedestrian_density_points_arrets_count_score",
        "pedestrian_density_terrasses_surface_raw",
        "pedestrian_density_terrasses_surface_score",
        "pedestrian_density_activities_2025_count_raw",
        "pedestrian_density_activities_2025_count_score",
        "pedestrian_density_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    qa_keep_columns = [
        "n_sq_qu",
        "c_qu",
        "c_quinsee",
        "l_qu",
        "c_ar",
        "n_sq_ar",
        "population_2022",
        "qa_surface_km2",
        "pedestrian_density_base_density_raw",
        "pedestrian_density_base_density_score",
        "pedestrian_density_tourist_sites_count_raw",
        "pedestrian_density_tourist_sites_count_score",
        "geometry",
    ]
    qa = qa[qa_keep_columns].copy()

    print(f"Writing {qa_output_path} ...")
    qa.to_file(qa_output_path, driver="GeoJSON")

    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pedestrian density on buffered Paris sidewalk sample points."
    )
    parser.add_argument(
        "--near_buffer_distance_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Buffer radius for sidewalk-adjacent features, in metres (25 ft)",
    )
    parser.add_argument(
        "--amenity_buffer_distance_m",
        type=float,
        default=AMENITY_BUFFER_DISTANCE_M,
        help="Buffer radius for schools, kiosks, municipal sites, and activities, in metres (200 ft)",
    )
    args = parser.parse_args()
    main(args)
