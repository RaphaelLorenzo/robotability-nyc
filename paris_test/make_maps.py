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


def plot_continuous_map(gdf, column, output_path, title, clamp_to_unit_interval=False):
    """Plot and save a continuous map with a colorbar."""
    if column not in gdf.columns:
        print(f"Skipping {column}, column not found.")
        return

    values = pd.to_numeric(gdf[column], errors="coerce")
    if values.notna().sum() == 0:
        print(f"Skipping {column}, no numeric values to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    is_point = gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all()
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
    else:
        plot_kwargs["linewidth"] = 0.15
        plot_kwargs["edgecolor"] = "none"
    if clamp_to_unit_interval:
        plot_kwargs["vmin"] = 0.0
        plot_kwargs["vmax"] = 1.0

    gdf.plot(**plot_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main(args):
    """Create the Paris sidewalk QA and pedestrian-density maps."""
    root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    processed_dir = os.path.join(root, "data", "processed")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    sidewalks_path = os.path.join(processed_dir, "sidewalks_paris.geojson")
    sidewalk_widths_path = os.path.join(processed_dir, "sidewalk_widths_paris.geojson")
    pedestrian_density_path = os.path.join(processed_dir, "pedestrian_density_paris.geojson")

    sidewalks = gpd.read_file(sidewalks_path)
    sidewalk_widths = gpd.read_file(sidewalk_widths_path)
    pedestrian_density = gpd.read_file(pedestrian_density_path) if os.path.exists(pedestrian_density_path) else None

    print(f"Reading sidewalks from {sidewalks_path} : got {sidewalks.shape[0]} rows")
    print(f"Reading sidewalk widths from {sidewalk_widths_path} : got {sidewalk_widths.shape[0]} rows")
    if pedestrian_density is not None:
        print(f"Reading pedestrian density from {pedestrian_density_path} : got {pedestrian_density.shape[0]} rows")

    plt.rc("font", family="serif")

    sidewalks = sidewalks.to_crs("EPSG:3857")
    sidewalk_widths = sidewalk_widths.to_crs("EPSG:3857")
    if pedestrian_density is not None:
        pedestrian_density = pedestrian_density.to_crs("EPSG:3857")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create simple Paris sidewalk maps.")
    args = parser.parse_args()
    main(args)
