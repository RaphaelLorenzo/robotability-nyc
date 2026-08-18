import argparse
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    """Plot processed sidewalks colored by tile and arrondissement."""
    root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    processed_dir = os.path.join(root, "data", "processed")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    sidewalks_path = os.path.join(processed_dir, "sidewalks_paris.geojson")
    sidewalk_widths_path = os.path.join(processed_dir, "sidewalk_widths_paris.geojson")
    sidewalks = gpd.read_file(sidewalks_path)
    sidewalk_widths = gpd.read_file(sidewalk_widths_path)

    plt.rc("font", family="serif")

    # Use Web Mercator so saved maps are easy to inspect visually.
    sidewalks = sidewalks.to_crs("EPSG:3857")
    sidewalk_widths = sidewalk_widths.to_crs("EPSG:3857")

    import numpy as np
    import matplotlib.colors as mcolors

    # Assign a random color to each unique pvp_tile
    unique_tiles = sorted(sidewalks["pvp_tile"].dropna().unique())
    rng = np.random.default_rng(42)  # seed for reproducibility

    color_map = {
        tile: mcolors.to_hex(rng.random(3)) for tile in unique_tiles
    }
    # Add NaN/missing if any
    color_map[np.nan] = "lightgrey"

    # Map sidewalk pvp_tile values to colors, including missing (NaN)
    def get_color(tile):
        if tile in color_map:
            return color_map[tile]
        elif pd.isna(tile):
            return "lightgrey"
        else:
            # assign a random color if it is truly unseen
            return mcolors.to_hex(rng.random(3))
    
    print(f"Number of unique PVP tiles: {len(unique_tiles)}")
    color_list = sidewalks["pvp_tile"].map(get_color)
    color_list_widths = sidewalk_widths["pvp_tile"].map(get_color)

    fig, ax = plt.subplots(figsize=(14, 14))
    sidewalks.plot(
        ax=ax,
        color=color_list,
        linewidth=0.15,
        edgecolor='none'
    )
    ax.set_title("Paris sidewalks colored by PVP tile")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "sidewalks_by_pvp_tile.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 14))
    sidewalks.plot(
        ax=ax,
        column="arrondissement",
        cmap="tab20",
        linewidth=0.15,
        categorical=True,
        legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 8},
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Paris sidewalks colored by arrondissement")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "sidewalks_by_arrondissement.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 14))
    sidewalk_widths.plot(
        ax=ax,
        color=color_list_widths,
        linewidth=0.35,
    )
    ax.set_title("Paris sidewalk width segments colored by PVP tile")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "sidewalk_widths_by_pvp_tile.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 14))
    sidewalk_widths.plot(
        ax=ax,
        column="arrondissement",
        cmap="tab20",
        linewidth=0.35,
        categorical=True,
        legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 8},
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Paris sidewalk width segments colored by arrondissement")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "sidewalk_widths_by_arrondissement.png"), dpi=250)
    plt.close(fig)

    print(f"Wrote {os.path.join(figures_dir, 'sidewalks_by_pvp_tile.png')}")
    print(f"Wrote {os.path.join(figures_dir, 'sidewalks_by_arrondissement.png')}")
    print(f"Wrote {os.path.join(figures_dir, 'sidewalk_widths_by_pvp_tile.png')}")
    print(f"Wrote {os.path.join(figures_dir, 'sidewalk_widths_by_arrondissement.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create simple Paris sidewalk maps.")
    args = parser.parse_args()
    main(args)
