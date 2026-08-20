import argparse
import os
import sys

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_categorical_map(gdf, output_path, title, **plot_kwargs):
    """Plot and save a categorical map without axes."""
    fig, ax = plt.subplots(figsize=(14, 14))
    gdf.plot(ax=ax, **plot_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_continuous_map(
    gdf, column, output_path, title, clamp_to_unit_interval=False, vmin=None, vmax=None
):
    """Plot and save a continuous map with a colorbar.

    When clamp_to_unit_interval is True, fixes vmin/vmax to [0, 1].
    When vmin/vmax are provided they override the default data range.
    """
    if column not in gdf.columns:
        print(f"Skipping {column}, column not found.")
        return

    values = pd.to_numeric(gdf[column], errors="coerce")
    if values.notna().sum() == 0:
        print(f"Skipping {column}, no numeric values to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    is_point = gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all()
    is_line = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()
    plot_kwargs = {
        "ax": ax,
        "column": column,
        "cmap": "viridis",
        "legend": True,
        "missing_kwds": {"color": "lightgrey", "label": "Missing"},
    }
    if is_point:
        plot_kwargs["markersize"] = 2
        plot_kwargs["linewidth"] = 0
    elif is_line:
        plot_kwargs["linewidth"] = 0.5
    else:
        plot_kwargs["linewidth"] = 0.15
        plot_kwargs["edgecolor"] = "none"
    if clamp_to_unit_interval:
        plot_kwargs["vmin"] = 0.0
        plot_kwargs["vmax"] = 1.0
    else:
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

    gdf.plot(**plot_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def selected_features(args):
    """Expand --features into the map groups that should be written."""
    if "all" in args.features:
        return {
            "sidewalks",
            "pedestrian_density",
            "crowd_dynamics",
            "surface_condition",
            "intersection_safety",
            "street_furniture_density",
            "curb_ramp_availability",
            "communication_infrastructure",
            "digital_map_existence",
            "gps_signal_strength",
            "vehicle_traffic",
            "sidewalk_width",
            "sidewalk_roughness",
            "traffic_management",
            "zoning_laws",
            "slope_gradient",
            "street_lighting",
            "bicycle_traffic",
            "charging_station_proximity",
            "surveillance_coverage",
            "bike_lane_availability",
            "shade",
        }
    return set(args.features)


def main(args):
    """Create the Paris sidewalk QA and pedestrian-density maps."""
    root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    processed_dir = os.path.join(root, "data", "processed")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    features = selected_features(args)

    sidewalks_path = os.path.join(processed_dir, "sidewalks_paris.geojson")
    sidewalk_widths_path = os.path.join(processed_dir, "sidewalk_widths_paris.geojson")
    pedestrian_density_path = os.path.join(processed_dir, "pedestrian_density_paris.geojson")
    crowd_dynamics_path = os.path.join(processed_dir, "crowd_dynamics_paris.geojson")
    surface_condition_path = os.path.join(processed_dir, "surface_condition_paris.geojson")
    intersection_safety_path = os.path.join(processed_dir, "intersection_safety_paris.geojson")
    street_furniture_density_path = os.path.join(processed_dir, "street_furniture_density_paris.geojson")
    curb_ramps_path = os.path.join(processed_dir, "curb_ramps_paris.geojson")
    curb_ramp_availability_path = os.path.join(processed_dir, "curb_ramp_availability_paris.geojson")
    traffic_management_path = os.path.join(processed_dir, "traffic_management_paris.geojson")
    zoning_regulation_path = os.path.join(processed_dir, "zoning_regulation_paris.geojson")
    zoning_laws_path = os.path.join(processed_dir, "zoning_laws_paris.geojson")
    slope_gradient_path = os.path.join(processed_dir, "slope_gradient_paris.geojson")
    street_lighting_path = os.path.join(processed_dir, "street_lighting_paris.geojson")
    bike_traffic_path = os.path.join(processed_dir, "bike_traffic_paris.geojson")
    bicycle_traffic_path = os.path.join(processed_dir, "bicycle_traffic_paris.geojson")
    charging_stations_path = os.path.join(processed_dir, "charging_stations_paris.geojson")
    charging_station_proximity_path = os.path.join(processed_dir, "charging_station_proximity_paris.geojson")
    bike_lane_path = os.path.join(processed_dir, "bike_lane_paris.geojson")
    bike_lane_availability_path = os.path.join(processed_dir, "bike_lane_availability_paris.geojson")
    shade_path = os.path.join(processed_dir, "shade_paris.geojson")
    vehicle_traffic_path = os.path.join(processed_dir, "vehicle_traffic_paris.geojson")
    sidewalk_width_path = os.path.join(processed_dir, "sidewalk_width_paris.geojson")
    communication_infrastructure_path = os.path.join(processed_dir, "communication_infrastructure_paris.geojson")
    digital_map_existence_path = os.path.join(processed_dir, "digital_map_existence_paris.geojson")
    gps_signal_strength_path = os.path.join(processed_dir, "gps_signal_strength_paris.geojson")
    sidewalk_roughness_path = os.path.join(processed_dir, "sidewalk_roughness_paris.geojson")
    surveillance_coverage_path = os.path.join(processed_dir, "surveillance_coverage_paris.geojson")

    sidewalks = None
    sidewalk_widths = None
    pedestrian_density = None
    crowd_dynamics = None
    surface_condition = None
    intersection_safety = None
    street_furniture_density = None
    curb_ramps = None
    curb_ramp_availability = None
    traffic_management = None
    zoning_regulation = None
    zoning_laws = None
    slope_gradient = None
    street_lighting = None
    bike_traffic = None
    bicycle_traffic = None
    charging_stations = None
    charging_station_proximity = None
    bike_lane = None
    bike_lane_availability = None
    shade = None
    vehicle_traffic = None
    sidewalk_width = None
    communication_infrastructure = None
    digital_map_existence = None
    gps_signal_strength = None
    sidewalk_roughness = None
    surveillance_coverage = None

    if "sidewalks" in features:
        sidewalks = gpd.read_file(sidewalks_path)
        sidewalk_widths = gpd.read_file(sidewalk_widths_path)
        print(f"Reading sidewalks from {sidewalks_path} : got {sidewalks.shape[0]} rows")
        print(f"Reading sidewalk widths from {sidewalk_widths_path} : got {sidewalk_widths.shape[0]} rows")
    if "pedestrian_density" in features and os.path.exists(pedestrian_density_path):
        pedestrian_density = gpd.read_file(pedestrian_density_path)
        print(f"Reading pedestrian density from {pedestrian_density_path} : got {pedestrian_density.shape[0]} rows")
    if "crowd_dynamics" in features and os.path.exists(crowd_dynamics_path):
        crowd_dynamics = gpd.read_file(crowd_dynamics_path)
        print(f"Reading crowd dynamics from {crowd_dynamics_path} : got {crowd_dynamics.shape[0]} rows")
    if "surface_condition" in features and os.path.exists(surface_condition_path):
        surface_condition = gpd.read_file(surface_condition_path)
        print(f"Reading surface condition from {surface_condition_path} : got {surface_condition.shape[0]} rows")
    if "intersection_safety" in features and os.path.exists(intersection_safety_path):
        intersection_safety = gpd.read_file(intersection_safety_path)
        print(f"Reading intersection safety from {intersection_safety_path} : got {intersection_safety.shape[0]} rows")
    if "street_furniture_density" in features and os.path.exists(street_furniture_density_path):
        street_furniture_density = gpd.read_file(street_furniture_density_path)
        print(
            "Reading street furniture density from "
            f"{street_furniture_density_path} : got {street_furniture_density.shape[0]} rows"
        )
    if "curb_ramps" in features and os.path.exists(curb_ramps_path):
        curb_ramps = gpd.read_file(curb_ramps_path)
        print(f"Reading curb ramps from {curb_ramps_path} : got {curb_ramps.shape[0]} rows")
    if "curb_ramp_availability" in features and os.path.exists(curb_ramp_availability_path):
        curb_ramp_availability = gpd.read_file(curb_ramp_availability_path)
        print(
            "Reading curb ramp availability from "
            f"{curb_ramp_availability_path} : got {curb_ramp_availability.shape[0]} rows"
        )
    if "traffic_management" in features and os.path.exists(traffic_management_path):
        traffic_management = gpd.read_file(traffic_management_path)
        print(f"Reading traffic management from {traffic_management_path} : got {traffic_management.shape[0]} rows")
    if "zoning_regulation" in features and os.path.exists(zoning_regulation_path):
        zoning_regulation = gpd.read_file(zoning_regulation_path)
        print(f"Reading zoning regulation from {zoning_regulation_path} : got {zoning_regulation.shape[0]} rows")
    if "zoning_laws" in features and os.path.exists(zoning_laws_path):
        zoning_laws = gpd.read_file(zoning_laws_path)
        print(f"Reading zoning laws from {zoning_laws_path} : got {zoning_laws.shape[0]} rows")
    if "slope_gradient" in features and os.path.exists(slope_gradient_path):
        slope_gradient = gpd.read_file(slope_gradient_path)
        print(f"Reading slope gradient from {slope_gradient_path} : got {slope_gradient.shape[0]} rows")
    if "street_lighting" in features and os.path.exists(street_lighting_path):
        street_lighting = gpd.read_file(street_lighting_path)
        print(f"Reading street lighting from {street_lighting_path} : got {street_lighting.shape[0]} rows")
    if "bike_traffic" in features and os.path.exists(bike_traffic_path):
        bike_traffic = gpd.read_file(bike_traffic_path)
        print(f"Reading bike traffic from {bike_traffic_path} : got {bike_traffic.shape[0]} rows")
    if "bicycle_traffic" in features and os.path.exists(bicycle_traffic_path):
        bicycle_traffic = gpd.read_file(bicycle_traffic_path)
        print(f"Reading bicycle traffic from {bicycle_traffic_path} : got {bicycle_traffic.shape[0]} rows")
    if "charging_stations" in features and os.path.exists(charging_stations_path):
        charging_stations = gpd.read_file(charging_stations_path)
        print(f"Reading charging stations from {charging_stations_path} : got {charging_stations.shape[0]} rows")
    if "charging_station_proximity" in features and os.path.exists(charging_station_proximity_path):
        charging_station_proximity = gpd.read_file(charging_station_proximity_path)
        print(
            "Reading charging station proximity from "
            f"{charging_station_proximity_path} : got {charging_station_proximity.shape[0]} rows"
        )
    if "bike_lane" in features and os.path.exists(bike_lane_path):
        bike_lane = gpd.read_file(bike_lane_path)
        print(f"Reading bike lane from {bike_lane_path} : got {bike_lane.shape[0]} rows")
    if "bike_lane_availability" in features and os.path.exists(bike_lane_availability_path):
        bike_lane_availability = gpd.read_file(bike_lane_availability_path)
        print(
            "Reading bike lane availability from "
            f"{bike_lane_availability_path} : got {bike_lane_availability.shape[0]} rows"
        )
    if "shade" in features and os.path.exists(shade_path):
        shade = gpd.read_file(shade_path)
        print(f"Reading shade from {shade_path} : got {shade.shape[0]} rows")
    if "vehicle_traffic" in features and os.path.exists(vehicle_traffic_path):
        vehicle_traffic = gpd.read_file(vehicle_traffic_path)
        print(f"Reading vehicle traffic from {vehicle_traffic_path} : got {vehicle_traffic.shape[0]} rows")
    if "sidewalk_width" in features and os.path.exists(sidewalk_width_path):
        sidewalk_width = gpd.read_file(sidewalk_width_path)
        print(f"Reading sidewalk width from {sidewalk_width_path} : got {sidewalk_width.shape[0]} rows")
    if "communication_infrastructure" in features and os.path.exists(communication_infrastructure_path):
        communication_infrastructure = gpd.read_file(communication_infrastructure_path)
        print(
            "Reading communication infrastructure from "
            f"{communication_infrastructure_path} : got {communication_infrastructure.shape[0]} rows"
        )
    if "digital_map_existence" in features and os.path.exists(digital_map_existence_path):
        digital_map_existence = gpd.read_file(digital_map_existence_path)
        print(f"Reading digital map existence from {digital_map_existence_path} : got {digital_map_existence.shape[0]} rows")
    if "gps_signal_strength" in features and os.path.exists(gps_signal_strength_path):
        gps_signal_strength = gpd.read_file(gps_signal_strength_path)
        print(f"Reading GPS signal strength from {gps_signal_strength_path} : got {gps_signal_strength.shape[0]} rows")
    if "sidewalk_roughness" in features and os.path.exists(sidewalk_roughness_path):
        sidewalk_roughness = gpd.read_file(sidewalk_roughness_path)
        print(f"Reading sidewalk roughness from {sidewalk_roughness_path} : got {sidewalk_roughness.shape[0]} rows")
    if "surveillance_coverage" in features and os.path.exists(surveillance_coverage_path):
        surveillance_coverage = gpd.read_file(surveillance_coverage_path)
        print(f"Reading surveillance coverage from {surveillance_coverage_path} : got {surveillance_coverage.shape[0]} rows")

    plt.rc("font", family="serif")

    if sidewalks is not None:
        sidewalks = sidewalks.to_crs("EPSG:3857")
        sidewalk_widths = sidewalk_widths.to_crs("EPSG:3857")
    if pedestrian_density is not None:
        pedestrian_density = pedestrian_density.to_crs("EPSG:3857")
    if crowd_dynamics is not None:
        crowd_dynamics = crowd_dynamics.to_crs("EPSG:3857")
    if surface_condition is not None:
        surface_condition = surface_condition.to_crs("EPSG:3857")
    if intersection_safety is not None:
        intersection_safety = intersection_safety.to_crs("EPSG:3857")
    if street_furniture_density is not None:
        street_furniture_density = street_furniture_density.to_crs("EPSG:3857")
    if curb_ramps is not None:
        curb_ramps = curb_ramps.to_crs("EPSG:3857")
    if curb_ramp_availability is not None:
        curb_ramp_availability = curb_ramp_availability.to_crs("EPSG:3857")
    if traffic_management is not None:
        traffic_management = traffic_management.to_crs("EPSG:3857")
    if zoning_regulation is not None:
        zoning_regulation = zoning_regulation.to_crs("EPSG:3857")
    if zoning_laws is not None:
        zoning_laws = zoning_laws.to_crs("EPSG:3857")
    if slope_gradient is not None:
        slope_gradient = slope_gradient.to_crs("EPSG:3857")
    if street_lighting is not None:
        street_lighting = street_lighting.to_crs("EPSG:3857")
    if bike_traffic is not None:
        bike_traffic = bike_traffic.to_crs("EPSG:3857")
    if bicycle_traffic is not None:
        bicycle_traffic = bicycle_traffic.to_crs("EPSG:3857")
    if charging_stations is not None:
        charging_stations = charging_stations.to_crs("EPSG:3857")
    if charging_station_proximity is not None:
        charging_station_proximity = charging_station_proximity.to_crs("EPSG:3857")
    if bike_lane is not None:
        bike_lane = bike_lane.to_crs("EPSG:3857")
    if bike_lane_availability is not None:
        bike_lane_availability = bike_lane_availability.to_crs("EPSG:3857")
    if shade is not None:
        shade = shade.to_crs("EPSG:3857")
    if vehicle_traffic is not None:
        vehicle_traffic = vehicle_traffic.to_crs("EPSG:3857")
    if sidewalk_width is not None:
        sidewalk_width = sidewalk_width.to_crs("EPSG:3857")
    if communication_infrastructure is not None:
        communication_infrastructure = communication_infrastructure.to_crs("EPSG:3857")
    if digital_map_existence is not None:
        digital_map_existence = digital_map_existence.to_crs("EPSG:3857")
    if gps_signal_strength is not None:
        gps_signal_strength = gps_signal_strength.to_crs("EPSG:3857")
    if sidewalk_roughness is not None:
        sidewalk_roughness = sidewalk_roughness.to_crs("EPSG:3857")
    if surveillance_coverage is not None:
        surveillance_coverage = surveillance_coverage.to_crs("EPSG:3857")

    if sidewalks is not None:
        unique_tiles = sorted(sidewalks["pvp_tile"].dropna().unique())
        unique_qas = sorted(sidewalks["qa_c_qu"].dropna().unique())
        rng = np.random.default_rng(42)
        pvp_color_map = {tile: mcolors.to_hex(rng.random(3)) for tile in unique_tiles}
        qa_color_map = {qa: mcolors.to_hex(rng.random(3)) for qa in unique_qas}

        def get_color(tile):
            """Map one PVP tile label to a display color."""
            if pd.isna(tile):
                return "lightgrey"
            if tile in pvp_color_map:
                return pvp_color_map[tile]
            return mcolors.to_hex(rng.random(3))

        def get_qa_color(qa):
            """Map one QA label to a display color."""
            if pd.isna(qa):
                return "lightgrey"
            if qa in qa_color_map:
                return qa_color_map[qa]
            return mcolors.to_hex(rng.random(3))

        print(f"Number of unique PVP tiles: {len(unique_tiles)}")
        print(f"Number of unique QA: {len(unique_qas)}")

        plot_categorical_map(
            sidewalks,
            os.path.join(figures_dir, "sidewalks_by_pvp_tile.png"),
            "Paris sidewalks colored by PVP tile",
            color=sidewalks["pvp_tile"].map(get_color),
            linewidth=0.15,
            edgecolor="none",
        )
        plot_categorical_map(
            sidewalks,
            os.path.join(figures_dir, "sidewalks_by_arrondissement.png"),
            "Paris sidewalks colored by arrondissement",
            column="arrondissement",
            cmap="tab20",
            linewidth=0.15,
            categorical=True,
            legend=True,
            legend_kwds={"loc": "lower left", "fontsize": 8},
            missing_kwds={"color": "lightgrey"},
        )
        plot_categorical_map(
            sidewalks,
            os.path.join(figures_dir, "sidewalks_by_qa.png"),
            "Paris sidewalks colored by Quartier Administratif",
            color=sidewalks["qa_c_qu"].map(get_qa_color),
            linewidth=0.15,
            edgecolor="none",
        )
        plot_categorical_map(
            sidewalk_widths,
            os.path.join(figures_dir, "sidewalk_widths_by_pvp_tile.png"),
            "Paris sidewalk width segments colored by PVP tile",
            color=sidewalk_widths["pvp_tile"].map(get_color),
            linewidth=0.35,
        )
        plot_categorical_map(
            sidewalk_widths,
            os.path.join(figures_dir, "sidewalk_widths_by_arrondissement.png"),
            "Paris sidewalk width segments colored by arrondissement",
            column="arrondissement",
            cmap="tab20",
            linewidth=0.35,
            categorical=True,
            legend=True,
            legend_kwds={"loc": "lower left", "fontsize": 8},
            missing_kwds={"color": "lightgrey"},
        )
        plot_categorical_map(
            sidewalk_widths,
            os.path.join(figures_dir, "sidewalk_widths_by_qa.png"),
            "Paris sidewalk width segments colored by Quartier Administratif",
            color=sidewalk_widths["qa_c_qu"].map(get_qa_color),
            linewidth=0.35,
        )

        print(f"Wrote {os.path.join(figures_dir, 'sidewalks_by_pvp_tile.png')}")
        print(f"Wrote {os.path.join(figures_dir, 'sidewalks_by_arrondissement.png')}")
        print(f"Wrote {os.path.join(figures_dir, 'sidewalks_by_qa.png')}")
        print(f"Wrote {os.path.join(figures_dir, 'sidewalk_widths_by_pvp_tile.png')}")
        print(f"Wrote {os.path.join(figures_dir, 'sidewalk_widths_by_arrondissement.png')}")
        print(f"Wrote {os.path.join(figures_dir, 'sidewalk_widths_by_qa.png')}")

        # raw width — clamp color range to 2.5%–99.5% quantiles to suppress outliers
        w_raw = pd.to_numeric(sidewalk_widths["width_m"], errors="coerce")
        raw_lo, raw_hi = w_raw.quantile(0.025), w_raw.quantile(0.995)
        plot_continuous_map(
            sidewalk_widths,
            "width_m",
            os.path.join(figures_dir, "sidewalk_width_raw.png"),
            "Sidewalk width (m)",
            vmin=raw_lo,
            vmax=raw_hi,
        )
        print(f"Wrote {os.path.join(figures_dir, 'sidewalk_width_raw.png')}")

        # 0-1 normalized with 2.5%-99.5% quantile clamping
        w = pd.to_numeric(sidewalk_widths["width_m"], errors="coerce")
        lo, hi = w.quantile(0.025), w.quantile(0.995)
        sidewalk_widths["width_m_score"] = ((w - lo) / (hi - lo)).clip(0.0, 1.0)
        plot_continuous_map(
            sidewalk_widths,
            "width_m_score",
            os.path.join(figures_dir, "sidewalk_width_score.png"),
            "Sidewalk width score (2.5%-99.5% clamped)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'sidewalk_width_score.png')}")

    if pedestrian_density is not None:
        component_specs = [
            ("pedestrian_density_base_density_raw", "pedestrian_density_base_density_score", "Base population density"),
            ("pedestrian_density_tourist_sites_count_raw", "pedestrian_density_tourist_sites_count_score", "Tourist sites count"),
            ("pedestrian_density_lieux_municipaux_count_raw", "pedestrian_density_lieux_municipaux_count_score", "Lieux municipaux count"),
            ("pedestrian_density_colleges_count_raw", "pedestrian_density_colleges_count_score", "Colleges count"),
            ("pedestrian_density_ecoles_elementaires_count_raw", "pedestrian_density_ecoles_elementaires_count_score", "Primary schools count"),
            ("pedestrian_density_ecoles_maternelles_count_raw", "pedestrian_density_ecoles_maternelles_count_score", "Maternelles count"),
            ("pedestrian_density_kiosques_de_presse_count_raw", "pedestrian_density_kiosques_de_presse_count_score", "Kiosques de presse count"),
            ("pedestrian_density_points_arrets_count_raw", "pedestrian_density_points_arrets_count_score", "Points d'arrets count"),
            ("pedestrian_density_terrasses_surface_raw", "pedestrian_density_terrasses_surface_score", "Terrasses and etalages surface"),
            ("pedestrian_density_activities_2025_count_raw", "pedestrian_density_activities_2025_count_score", "Activities in 2025 count"),
        ]

        for raw_column, score_column, title_prefix in component_specs:
            plot_continuous_map(
                pedestrian_density,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
                clamp_to_unit_interval=False,
            )
            plot_continuous_map(
                pedestrian_density,
                score_column,
                os.path.join(figures_dir, f"{score_column}.png"),
                f"{title_prefix} score (2.5%-99.5% clamped)",
                clamp_to_unit_interval=True,
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            pedestrian_density,
            "pedestrian_density_score",
            os.path.join(figures_dir, "pedestrian_density_score.png"),
            "Pedestrian density score",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'pedestrian_density_score.png')}")

    if crowd_dynamics is not None:
        crowd_specs = [
            ("crowd_dynamics_zti_score", None, "ZTI membership (0.5 inside zone)"),
            ("crowd_dynamics_tourist_sites_count_raw", "crowd_dynamics_tourist_sites_count_score", "Tourist sites count"),
            ("crowd_dynamics_tourism_score", None, "Tourism purpose score before invert"),
        ]
        for raw_column, score_column, title_prefix in crowd_specs:
            plot_continuous_map(
                crowd_dynamics,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
                clamp_to_unit_interval=("score" in raw_column),
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            if score_column is not None:
                plot_continuous_map(
                    crowd_dynamics,
                    score_column,
                    os.path.join(figures_dir, f"{score_column}.png"),
                    f"{title_prefix} score (2.5%-99.5% clamped)",
                    clamp_to_unit_interval=True,
                )
                print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            crowd_dynamics,
            "crowd_dynamics_score",
            os.path.join(figures_dir, "crowd_dynamics_score.png"),
            "Crowd dynamics score (inverted, NYC polarity)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'crowd_dynamics_score.png')}")

    if surface_condition is not None:
        surface_specs = [
            ("surface_condition_chantier_flag", None, "Chantier occupancy of trottoir or piste cyclable"),
            ("surface_condition_anomaly_count_raw", "surface_condition_anomaly_count_score", "Dans Ma Rue surface anomalies"),
        ]
        for raw_column, score_column, title_prefix in surface_specs:
            plot_continuous_map(
                surface_condition,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
                clamp_to_unit_interval=("flag" in raw_column or "score" in raw_column),
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            if score_column is not None:
                plot_continuous_map(
                    surface_condition,
                    score_column,
                    os.path.join(figures_dir, f"{score_column}.png"),
                    f"{title_prefix} score (2.5%-99.5% clamped)",
                    clamp_to_unit_interval=True,
                )
                print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            surface_condition,
            "surface_condition_score",
            os.path.join(figures_dir, "surface_condition_score.png"),
            "Surface condition score (1 = good)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'surface_condition_score.png')}")

    if intersection_safety is not None:
        intersection_specs = [
            ("intersection_safety_aire_pietonne_flag", None, "Aire pietonne membership"),
            ("intersection_safety_zone_rencontre_flag", None, "Zone de rencontre membership"),
            ("intersection_safety_zone_score", None, "Calmed-zone base score (0.75 / 0.9 / 1.0)"),
            ("intersection_safety_accident_rate_raw", "intersection_safety_accident_rate_score", "Accidents per km2 by arrondissement"),
            ("intersection_safety_anomaly_count_raw", "intersection_safety_anomaly_count_score", "Dans Ma Rue intersection-safety anomalies"),
        ]
        for raw_column, score_column, title_prefix in intersection_specs:
            plot_continuous_map(
                intersection_safety,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
                clamp_to_unit_interval=("flag" in raw_column or "score" in raw_column),
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            if score_column is not None:
                plot_continuous_map(
                    intersection_safety,
                    score_column,
                    os.path.join(figures_dir, f"{score_column}.png"),
                    f"{title_prefix} score",
                    clamp_to_unit_interval=True,
                )
                print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            intersection_safety,
            "intersection_safety_score",
            os.path.join(figures_dir, "intersection_safety_score.png"),
            "Intersection safety score (1 = very safe)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'intersection_safety_score.png')}")

    if street_furniture_density is not None:
        street_furniture_specs = [
            (
                "street_furniture_jardinieres_bancs_corbeilles_count_raw",
                "street_furniture_jardinieres_bancs_corbeilles_count_score",
                "Jardinieres, bancs, and corbeilles",
            ),
            (
                "street_furniture_bornes_barrieres_potelets_count_raw",
                "street_furniture_bornes_barrieres_potelets_count_score",
                "Bornes, barrieres, and potelets",
            ),
            (
                "street_furniture_kiosques_toilettes_panneaux_count_raw",
                "street_furniture_kiosques_toilettes_panneaux_count_score",
                "Kiosques, toilettes, and panneaux",
            ),
            (
                "street_furniture_composteurs_count_raw",
                "street_furniture_composteurs_count_score",
                "Composteurs",
            ),
            (
                "street_furniture_trilib_count_raw",
                "street_furniture_trilib_count_score",
                "Trilib stations",
            ),
            (
                "street_furniture_fontaines_count_raw",
                "street_furniture_fontaines_count_score",
                "Available Paris drinking fountains",
            ),
            (
                "street_furniture_anomaly_count_raw",
                "street_furniture_anomaly_count_score",
                "Dans Ma Rue street-furniture anomalies",
            ),
        ]
        for raw_column, score_column, title_prefix in street_furniture_specs:
            plot_continuous_map(
                street_furniture_density,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
                clamp_to_unit_interval=("score" in raw_column),
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            if score_column is not None:
                plot_continuous_map(
                    street_furniture_density,
                    score_column,
                    os.path.join(figures_dir, f"{score_column}.png"),
                    f"{title_prefix} score (2.5%-99.5% clamped)",
                    clamp_to_unit_interval=True,
                )
                print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            street_furniture_density,
            "street_furniture_density_score",
            os.path.join(figures_dir, "street_furniture_density_score.png"),
            "Street furniture density score (1 = less clutter)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'street_furniture_density_score.png')}")

    if curb_ramps is not None:
        for flag_column, title_prefix in [
            ("curb_ramps_escalier_flag", "Voie en escalier"),
            ("curb_ramps_accessibilite_flag", "Quartier d'accessibilité augmentée"),
        ]:
            plot_continuous_map(
                curb_ramps,
                flag_column,
                os.path.join(figures_dir, f"{flag_column}.png"),
                f"{title_prefix} flag",
                clamp_to_unit_interval=True,
            )
            print(f"Wrote {os.path.join(figures_dir, f'{flag_column}.png')}")

        plot_continuous_map(
            curb_ramps,
            "curb_ramps_score",
            os.path.join(figures_dir, "curb_ramps_score.png"),
            "Curb-ramp availability score (0 = stair street, 0.75 = default, 0.9 = accessibility quarter)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'curb_ramps_score.png')}")

    if traffic_management is not None:
        for raw_column, score_column, title_prefix in [
            ("traffic_management_feux_count_raw", "traffic_management_feux_count_score", "Feux tricolores count"),
            ("traffic_management_anomaly_count_raw", "traffic_management_anomaly_count_score", "Dans Ma Rue traffic-management anomalies"),
        ]:
            plot_continuous_map(
                traffic_management,
                raw_column,
                os.path.join(figures_dir, f"{raw_column}.png"),
                f"{title_prefix} per sidewalk sample point",
            )
            print(f"Wrote {os.path.join(figures_dir, f'{raw_column}.png')}")
            plot_continuous_map(
                traffic_management,
                score_column,
                os.path.join(figures_dir, f"{score_column}.png"),
                f"{title_prefix} score (2.5%-99.5% clamped)",
                clamp_to_unit_interval=True,
            )
            print(f"Wrote {os.path.join(figures_dir, f'{score_column}.png')}")

        plot_continuous_map(
            traffic_management,
            "traffic_management_score",
            os.path.join(figures_dir, "traffic_management_score.png"),
            "Traffic management score (0.5 base + feux bonus − anomaly malus)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'traffic_management_score.png')}")

    if zoning_regulation is not None:
        for flag_column, title_prefix in [
            ("zoning_regulation_aire_pietonne_flag", "Aire piétonne"),
            ("zoning_regulation_zone_rencontre_flag", "Zone de rencontre"),
            ("zoning_regulation_ztl_flag", "Zone à Trafic Limité (ZTL)"),
            ("zoning_regulation_paris_respire_flag", "Paris Respire secteur"),
        ]:
            plot_continuous_map(
                zoning_regulation,
                flag_column,
                os.path.join(figures_dir, f"{flag_column}.png"),
                f"{title_prefix} flag",
                clamp_to_unit_interval=True,
            )
            print(f"Wrote {os.path.join(figures_dir, f'{flag_column}.png')}")

        plot_continuous_map(
            zoning_regulation,
            "zoning_regulation_score",
            os.path.join(figures_dir, "zoning_regulation_score.png"),
            "Zoning regulation score (0.3 base → 0.7 calmed zone + ZTL/Respire bonuses)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'zoning_regulation_score.png')}")

    if slope_gradient is not None:
        plot_continuous_map(
            slope_gradient,
            "slope_gradient_elevation_m",
            os.path.join(figures_dir, "slope_gradient_elevation_m.png"),
            "Assigned elevation (m) per sidewalk sample point",
        )
        print(f"Wrote {os.path.join(figures_dir, 'slope_gradient_elevation_m.png')}")
        plot_continuous_map(
            slope_gradient,
            "slope_gradient_mean_raw",
            os.path.join(figures_dir, "slope_gradient_mean_raw.png"),
            "Mean slope to neighbors (rise/run) per sidewalk sample point",
        )
        print(f"Wrote {os.path.join(figures_dir, 'slope_gradient_mean_raw.png')}")
        plot_continuous_map(
            slope_gradient,
            "slope_gradient_score",
            os.path.join(figures_dir, "slope_gradient_score.png"),
            "Slope gradient score (1 = steep, 0 = flat)",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {os.path.join(figures_dir, 'slope_gradient_score.png')}")

    if street_lighting is not None:
        plot_continuous_map(
            street_lighting, "street_lighting_lamp_count_raw",
            os.path.join(figures_dir, "street_lighting_lamp_count_raw.png"),
            "Public lamp count per sidewalk sample point",
        )
        plot_continuous_map(
            street_lighting, "street_lighting_score",
            os.path.join(figures_dir, "street_lighting_score.png"),
            "Street lighting score (2.5%-99.5% clamped)", clamp_to_unit_interval=True,
        )
        print("Wrote street_lighting maps")

    if bike_traffic is not None:
        plot_continuous_map(
            bike_traffic, "bike_traffic_piste_flag",
            os.path.join(figures_dir, "bike_traffic_piste_flag.png"),
            "Adjacent piste cyclable / couloir mixte flag", clamp_to_unit_interval=True,
        )
        plot_continuous_map(
            bike_traffic, "bike_traffic_velib_count_raw",
            os.path.join(figures_dir, "bike_traffic_velib_count_raw.png"),
            "Vélib stations count (200 ft)",
        )
        plot_continuous_map(
            bike_traffic, "bike_traffic_score",
            os.path.join(figures_dir, "bike_traffic_score.png"),
            "Bicycle traffic score (0.8 piste + 0.2 Vélib)", clamp_to_unit_interval=True,
        )
        print("Wrote bike_traffic maps")

    if charging_stations is not None:
        plot_continuous_map(
            charging_stations, "charging_stations_count_raw",
            os.path.join(figures_dir, "charging_stations_count_raw.png"),
            "Vélib stations count (200 ft)",
        )
        plot_continuous_map(
            charging_stations, "charging_stations_score",
            os.path.join(figures_dir, "charging_stations_score.png"),
            "Charging stations score (2.5%-99.5% clamped)", clamp_to_unit_interval=True,
        )
        print("Wrote charging_stations maps")

    if bike_lane is not None:
        plot_continuous_map(
            bike_lane, "bike_lane_score",
            os.path.join(figures_dir, "bike_lane_score.png"),
            "Bike lane availability score (0 / 0.8 piste / 1.0 piste cyclable)",
            clamp_to_unit_interval=True,
        )
        print("Wrote bike_lane maps")

    if shade is not None:
        plot_continuous_map(
            shade, "shade_tree_count_raw",
            os.path.join(figures_dir, "shade_tree_count_raw.png"),
            "Weighted tree count (50 ft, young=0.5)",
        )
        plot_continuous_map(
            shade, "shade_score",
            os.path.join(figures_dir, "shade_score.png"),
            "Shade score (2.5%-99.5% clamped)", clamp_to_unit_interval=True,
        )
        print("Wrote shade maps")

    if sidewalk_width is not None:
        plot_continuous_map(
            sidewalk_width, "sidewalk_width_raw",
            os.path.join(figures_dir, "sidewalk_width_feature_raw.png"),
            "Sidewalk width raw (m)",
        )
        plot_continuous_map(
            sidewalk_width, "sidewalk_width_score",
            os.path.join(figures_dir, "sidewalk_width_feature_score.png"),
            "Sidewalk width feature score (2.5%-99.5% clamped)",
            clamp_to_unit_interval=True,
        )
        print("Wrote sidewalk_width feature maps")

    if vehicle_traffic is not None:
        plot_continuous_map(
            vehicle_traffic, "vehicle_traffic_occupation_raw",
            os.path.join(figures_dir, "vehicle_traffic_occupation_raw.png"),
            "Vehicle traffic occupation rate on 2026-06-01",
        )
        plot_continuous_map(
            vehicle_traffic, "vehicle_traffic_score",
            os.path.join(figures_dir, "vehicle_traffic_score.png"),
            "Vehicle traffic score (2.5%-99.5% clamped)",
            clamp_to_unit_interval=True,
        )
        print("Wrote vehicle_traffic maps")

    if curb_ramp_availability is not None:
        plot_continuous_map(
            curb_ramp_availability, "curb_ramp_availability_score",
            os.path.join(figures_dir, "curb_ramp_availability_score.png"),
            "Curb ramp availability score",
            clamp_to_unit_interval=True,
        )
        print("Wrote curb_ramp_availability score map")

    if zoning_laws is not None:
        plot_continuous_map(
            zoning_laws, "zoning_laws_score",
            os.path.join(figures_dir, "zoning_laws_score.png"),
            "Zoning laws score",
            clamp_to_unit_interval=True,
        )
        print("Wrote zoning_laws score map")

    if bicycle_traffic is not None:
        plot_continuous_map(
            bicycle_traffic, "bicycle_traffic_score",
            os.path.join(figures_dir, "bicycle_traffic_score.png"),
            "Bicycle traffic score",
            clamp_to_unit_interval=True,
        )
        print("Wrote bicycle_traffic score map")

    if charging_station_proximity is not None:
        plot_continuous_map(
            charging_station_proximity, "charging_station_proximity_score",
            os.path.join(figures_dir, "charging_station_proximity_score.png"),
            "Charging station proximity score",
            clamp_to_unit_interval=True,
        )
        print("Wrote charging_station_proximity score map")

    if bike_lane_availability is not None:
        plot_continuous_map(
            bike_lane_availability, "bike_lane_availability_score",
            os.path.join(figures_dir, "bike_lane_availability_score.png"),
            "Bike lane availability score",
            clamp_to_unit_interval=True,
        )
        print("Wrote bike_lane_availability score map")

    for feature_gdf, feature_name in [
        (communication_infrastructure, "communication_infrastructure"),
        (digital_map_existence, "digital_map_existence"),
        (gps_signal_strength, "gps_signal_strength"),
        (sidewalk_roughness, "sidewalk_roughness"),
        (surveillance_coverage, "surveillance_coverage"),
    ]:
        if feature_gdf is None:
            continue
        plot_continuous_map(
            feature_gdf,
            f"{feature_name}_score",
            os.path.join(figures_dir, f"{feature_name}_score.png"),
            f"{feature_name} score",
            clamp_to_unit_interval=True,
        )
        print(f"Wrote {feature_name} score map")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create simple Paris sidewalk maps.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "sidewalks",
            "pedestrian_density",
            "crowd_dynamics",
            "surface_condition",
            "intersection_safety",
            "street_furniture_density",
            "curb_ramps",
            "curb_ramp_availability",
            "communication_infrastructure",
            "digital_map_existence",
            "gps_signal_strength",
            "vehicle_traffic",
            "sidewalk_width",
            "sidewalk_roughness",
            "traffic_management",
            "zoning_regulation",
            "zoning_laws",
            "slope_gradient",
            "street_lighting",
            "bike_traffic",
            "bicycle_traffic",
            "charging_stations",
            "charging_station_proximity",
            "surveillance_coverage",
            "bike_lane",
            "bike_lane_availability",
            "shade",
        ],
        help="Which map groups to write (default: all)",
    )
    args = parser.parse_args()
    main(args)
