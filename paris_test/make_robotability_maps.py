import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = "EPSG:2154"
LEVELS = ["point", "segment", "qa", "pvp", "arrondissement"]


def min_max_01(series):
    """Rescale a numeric series to [0, 1]; constant series becomes 0."""
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - lo) / (hi - lo)


def plot_score_01(gdf, column, output_path, title):
    """Continuous viridis map fixed to the [0, 1] interval."""
    fig, ax = plt.subplots(figsize=(14, 14))
    is_point = gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all()
    is_line = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()
    plot_kwargs = {
        "ax": ax,
        "column": column,
        "cmap": "viridis",
        "legend": True,
        "vmin": 0.0,
        "vmax": 1.0,
        "missing_kwds": {"color": "lightgrey", "label": "Missing"},
    }
    if is_point:
        plot_kwargs["markersize"] = 2
        plot_kwargs["linewidth"] = 0
    elif is_line:
        plot_kwargs["linewidth"] = 0.5
    else:
        plot_kwargs["linewidth"] = 0.1
        plot_kwargs["edgecolor"] = "0.4"
    gdf.plot(**plot_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_score_quantiles(gdf, column, output_path, title, n_quantiles):
    """Choropleth colored by quantile bins of the score column."""
    values = pd.to_numeric(gdf[column], errors="coerce")
    valid = values.dropna()
    if len(valid) == 0:
        print(f"Skipping quantiles for {column}: no numeric values.")
        return

    # qcut may drop duplicate edges on highly discrete data.
    categories, bins = pd.qcut(
        valid, q=n_quantiles, labels=False, retbins=True, duplicates="drop"
    )
    plot_df = gdf.copy()
    n_bins = len(bins) - 1
    labels = [f"Q{i + 1}: {bins[i]:.3f}–{bins[i + 1]:.3f}" for i in range(n_bins)]
    plot_df["_quantile"] = pd.Series(pd.NA, index=plot_df.index, dtype="object")
    plot_df.loc[valid.index, "_quantile"] = [labels[int(i)] for i in categories]
    plot_df["_quantile"] = pd.Categorical(plot_df["_quantile"], categories=labels, ordered=True)

    fig, ax = plt.subplots(figsize=(14, 14))
    is_point = plot_df.geometry.geom_type.isin(["Point", "MultiPoint"]).all()
    is_line = plot_df.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()
    plot_kwargs = {
        "ax": ax,
        "column": "_quantile",
        "cmap": "viridis",
        "categorical": True,
        "legend": True,
        "legend_kwds": {"title": "Quantile", "loc": "lower left"},
        "missing_kwds": {"color": "lightgrey", "label": "Missing"},
    }
    if is_point:
        plot_kwargs["markersize"] = 2
        plot_kwargs["linewidth"] = 0
    elif is_line:
        plot_kwargs["linewidth"] = 0.5
    else:
        plot_kwargs["linewidth"] = 0.1
        plot_kwargs["edgecolor"] = "0.4"
    plot_df.plot(**plot_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def aggregate_mean_score(points, group_col):
    """Mean robotability_score by group key, dropping null keys."""
    table = points[[group_col, "robotability_score"]].dropna(subset=[group_col]).copy()
    table[group_col] = table[group_col].astype(str)
    grouped = (
        table.groupby(group_col, dropna=False)["robotability_score"]
        .mean()
        .rename("robotability_score")
        .reset_index()
    )
    grouped["robotability_score_01"] = min_max_01(grouped["robotability_score"])
    return grouped


def load_points(score_path):
    """Load per-point robotability scores into projected CRS."""
    print(f"Loading scores from {score_path} ...")
    points = gpd.read_file(score_path).to_crs(PROJ)
    print(f"Loaded {len(points)} points")
    return points


def geometry_for_level(level, root, points):
    """Build a GeoDataFrame of geometries with aggregated scores for one level."""
    processed = os.path.join(root, "data", "processed")
    raw = os.path.join(root, "data", "raw")

    if level == "point":
        gdf = points.copy()
        if "robotability_score_01" not in gdf.columns:
            gdf["robotability_score_01"] = min_max_01(gdf["robotability_score"])
        return gdf

    if level == "segment":
        # Paris segment_index == sidewalk_id; fetch original sidewalk polygons.
        sidewalks_path = os.path.join(processed, "sidewalks_paris.geojson")
        print(f"Loading sidewalk segments from {sidewalks_path} ...")
        sidewalks = gpd.read_file(sidewalks_path).to_crs(PROJ)
        sidewalks["sidewalk_id"] = sidewalks["sidewalk_id"].astype(str)
        scores = aggregate_mean_score(points, "sidewalk_id")
        return sidewalks.merge(scores, on="sidewalk_id", how="inner")

    if level == "qa":
        qa_path = os.path.join(raw, "quartier_paris_raw.geojson")
        print(f"Loading quartiers administratifs from {qa_path} ...")
        qa = gpd.read_file(qa_path).to_crs(PROJ)
        qa["c_qu"] = qa["c_qu"].astype(str)
        scores = aggregate_mean_score(points, "qa_c_qu").rename(columns={"qa_c_qu": "c_qu"})
        return qa.merge(scores, on="c_qu", how="inner")

    if level == "pvp":
        pvp_path = os.path.join(raw, "pvp_tiles_raw.geojson")
        print(f"Loading PVP tiles from {pvp_path} ...")
        pvp = gpd.read_file(pvp_path).to_crs(PROJ)
        pvp["numero_pave"] = pvp["numero_pave"].astype(str)
        scores = aggregate_mean_score(points, "pvp_tile").rename(columns={"pvp_tile": "numero_pave"})
        return pvp.merge(scores, on="numero_pave", how="inner")

    if level == "arrondissement":
        ar_path = os.path.join(raw, "arrondissements_raw.geojson")
        print(f"Loading arrondissements from {ar_path} ...")
        ar = gpd.read_file(ar_path).to_crs(PROJ)
        ar["c_ar"] = ar["c_ar"].astype(str)
        # Prefer numeric arrondissement id when present, else qa_c_ar / label.
        if "arrondissement_id" in points.columns and points["arrondissement_id"].notna().any():
            key = "arrondissement_id"
        elif "qa_c_ar" in points.columns:
            key = "qa_c_ar"
        else:
            key = "arrondissement"
        scores = aggregate_mean_score(points, key)
        scores = scores.rename(columns={key: "c_ar"})
        scores["c_ar"] = scores["c_ar"].astype(str).str.replace(r"\.0$", "", regex=True)
        ar["c_ar"] = ar["c_ar"].astype(str).str.replace(r"\.0$", "", regex=True)
        return ar.merge(scores, on="c_ar", how="inner")

    raise ValueError(f"Unknown level: {level}")


def write_maps_for_level(gdf, level, figures_dir, n_quantiles):
    """Write the 0-1 and quantile maps for one aggregation level."""
    if len(gdf) == 0:
        print(f"No geometries for level={level}, skipping.")
        return

    score_01_path = os.path.join(figures_dir, f"robotability_score_{level}_01.png")
    quantiles_path = os.path.join(figures_dir, f"robotability_score_{level}_quantiles.png")

    print(f"Writing {score_01_path} ({len(gdf)} features) ...")
    plot_score_01(
        gdf,
        "robotability_score_01",
        score_01_path,
        f"Paris robotability score (0–1) — {level}",
    )

    print(f"Writing {quantiles_path} ...")
    plot_score_quantiles(
        gdf,
        "robotability_score",
        quantiles_path,
        f"Paris robotability score (quantiles) — {level}",
        n_quantiles=n_quantiles,
    )


def selected_levels(args):
    """Expand --level into the concrete levels to plot."""
    if "all" in args.level:
        return list(LEVELS)
    return list(dict.fromkeys(args.level))


def main(args):
    """Map Paris robotability scores at point / segment / QA / PVP / arrondissement."""
    root = os.path.dirname(os.path.abspath(__file__))
    processed = os.path.join(root, "data", "processed")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    score_path = args.score_path or os.path.join(processed, "robotability_score_paris.geojson")
    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f"Missing {score_path}. Run compute_robotability_score.py first."
        )

    points = load_points(score_path)
    levels = selected_levels(args)
    print(f"Levels: {levels}")

    for level in levels:
        print(f"\n=== {level} ===")
        gdf = geometry_for_level(level, root, points)
        write_maps_for_level(gdf, level, figures_dir, args.n_quantiles)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map Paris robotability scores at several geographic levels."
    )
    parser.add_argument(
        "--level",
        nargs="+",
        default=["point"],
        choices=LEVELS + ["all"],
        help="Aggregation level(s) to map (default: point). Use 'all' for every level.",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=5,
        help="Number of quantile bins for the quantile maps (default: 5).",
    )
    parser.add_argument(
        "--score_path",
        default=None,
        help="Optional path to robotability_score_paris.geojson",
    )
    args = parser.parse_args()
    main(args)
