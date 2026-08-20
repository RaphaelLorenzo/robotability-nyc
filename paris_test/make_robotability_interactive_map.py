"""Build a MapLibre interactive robotability map for Paris.

Levels: sidewalk_full, sidewalk_segmentized, qa, pvp, arrondissement.
Each feature is colored by robotability_score_01; click opens a panel with every
CSV column as raw (aggregated) value and a level-wise 0-1 min-max normalization.
"""

from __future__ import annotations

import argparse
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

PROJ = "EPSG:2154"
WGS = "EPSG:4326"

LEVELS = [
    "sidewalk_full",
    "sidewalk_segmentized",
    "qa",
    "pvp",
    "arrondissement",
]

# Columns kept as identifiers / labels (first value on aggregate, not mean).
ID_COLUMNS = {
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
    "iu_ac",
    "lon",
    "lat",
}


def min_max_01(series):
    """Rescale a numeric series to [0, 1]; constant / empty series becomes 0."""
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - lo) / (hi - lo)


def json_safe(value):
    """Convert pandas/numpy values to JSON-serializable Python objects."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def format_number(value, digits=4):
    """Round floats for compact GeoJSON properties."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def load_features_table(csv_path):
    """Load the robotability CSV and classify columns."""
    print(f"Loading feature table from {csv_path} ...")
    table = pd.read_csv(csv_path)
    print(f"Loaded {len(table):,} rows × {len(table.columns)} columns")

    value_columns = list(table.columns)
    skip_numeric = ID_COLUMNS - {"width_m"}
    numeric_columns = []
    for col in value_columns:
        if col in skip_numeric:
            continue
        converted = pd.to_numeric(table[col], errors="coerce")
        if converted.notna().mean() >= 0.5:
            table[col] = converted
            numeric_columns.append(col)

    return table, value_columns, numeric_columns


def aggregate_table(table, group_col, value_columns, numeric_columns):
    """Mean-aggregate numeric columns by group_col; keep first for id columns."""
    work = table.dropna(subset=[group_col]).copy()
    work[group_col] = work[group_col].astype(str).str.replace(r"\.0$", "", regex=True)

    agg_spec = {}
    for col in value_columns:
        if col == group_col:
            continue
        if col in numeric_columns:
            agg_spec[col] = "mean"
        else:
            agg_spec[col] = "first"

    grouped = work.groupby(group_col, dropna=False).agg(agg_spec).reset_index()
    return grouped


def attach_norm01(df, numeric_columns):
    """Add parallel *_n01 columns (min-max within this layer)."""
    out = df.copy()
    for col in numeric_columns:
        if col not in out.columns:
            continue
        # Preserve existing robotability_score_01 from the CSV when present at point
        # level; still recompute a layer-wise sibling under the uniform *_n01 scheme.
        out[f"{col}__n01"] = min_max_01(out[col])
    return out


def simplify_polygons(gdf, tolerance_m):
    """Simplify polygon geometries in a projected CRS, then return WGS84."""
    if tolerance_m <= 0:
        return gdf.to_crs(WGS)
    projected = gdf.to_crs(PROJ)
    projected["geometry"] = projected.geometry.simplify(tolerance_m, preserve_topology=True)
    return projected.to_crs(WGS)


def write_level(out_dir, level, gdf, feature_ids, feature_labels, value_columns, numeric_columns):
    """Write a light GeoJSON (for map paint) plus a columnar attributes JSON (for clicks)."""
    score_01_col = (
        "robotability_score__n01"
        if "robotability_score__n01" in gdf.columns
        else "robotability_score_01"
    )
    present = [c for c in value_columns if c in gdf.columns]
    n01_present = [c for c in numeric_columns if f"{c}__n01" in gdf.columns]

    # Keep geometry and attribute rows aligned (drop empty geoms once).
    work = gdf.copy()
    work["_feature_id"] = [str(x) for x in feature_ids]
    work["_feature_label"] = [str(x) for x in feature_labels]
    work = work.loc[work.geometry.notna() & ~work.geometry.is_empty].copy()

    features = []
    for geom, fid, label, score, score_01 in zip(
        work.geometry,
        work["_feature_id"],
        work["_feature_label"],
        work["robotability_score"] if "robotability_score" in work.columns else [None] * len(work),
        work[score_01_col] if score_01_col in work.columns else [None] * len(work),
    ):
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "feature_id": fid,
                    "feature_label": label,
                    "robotability_score": format_number(json_safe(score)),
                    "robotability_score_01": format_number(json_safe(score_01)),
                },
            }
        )

    geo_path = os.path.join(out_dir, f"{level}.geojson")
    with open(geo_path, "w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, separators=(",", ":"))
    geo_mb = os.path.getsize(geo_path) / (1024 * 1024)

    raw = {col: [format_number(json_safe(v)) for v in work[col].tolist()] for col in present}
    n01 = {}
    for col in present:
        if col in n01_present:
            n01[col] = [format_number(json_safe(v)) for v in work[f"{col}__n01"].tolist()]
        else:
            n01[col] = [None] * len(work)

    attrs = {
        "columns": present,
        "ids": work["_feature_id"].tolist(),
        "labels": work["_feature_label"].tolist(),
        "raw": raw,
        "n01": n01,
    }
    attrs_path = os.path.join(out_dir, f"{level}.attrs.json")
    with open(attrs_path, "w", encoding="utf-8") as handle:
        json.dump(attrs, handle, separators=(",", ":"))
    attrs_mb = os.path.getsize(attrs_path) / (1024 * 1024)
    print(
        f"  wrote {geo_path} ({len(features):,} features, {geo_mb:.1f} MB) and "
        f"{os.path.basename(attrs_path)} ({attrs_mb:.1f} MB)"
    )
    return len(features)


def build_sidewalk_full(root, table, value_columns, numeric_columns, out_dir, simplify_m):
    """Aggregate sample points to full sidewalk polygons."""
    processed = os.path.join(root, "data", "processed")
    sidewalks_path = os.path.join(processed, "sidewalks_paris.geojson")
    print(f"Building sidewalk_full from {sidewalks_path} ...")
    sidewalks = gpd.read_file(sidewalks_path)
    sidewalks["sidewalk_id"] = sidewalks["sidewalk_id"].astype(str)
    scores = aggregate_table(table, "sidewalk_id", value_columns, numeric_columns)
    scores = attach_norm01(scores, numeric_columns)
    merged = sidewalks.merge(scores, on="sidewalk_id", how="inner")
    merged = simplify_polygons(merged, simplify_m)
    ids = merged["sidewalk_id"].astype(str).tolist()
    labels = [f"sidewalk {sid}" for sid in ids]
    return write_level(
        out_dir, "sidewalk_full", merged, ids, labels, value_columns, numeric_columns
    )


def build_sidewalk_segmentized(
    root, table, value_columns, numeric_columns, out_dir, max_points
):
    """Write segmentized sample points (optionally subsampled) as a GeoJSON layer."""
    print("Building sidewalk_segmentized point layer ...")
    work = table.copy()
    if max_points is not None and max_points > 0 and len(work) > max_points:
        print(f"  subsampling {len(work):,} → {max_points:,} points")
        work = work.sample(n=max_points, random_state=0).sort_values("point_index")
    work = attach_norm01(work, numeric_columns)

    geometry = gpd.points_from_xy(work["lon"], work["lat"], crs=WGS)
    gdf = gpd.GeoDataFrame(work, geometry=geometry, crs=WGS)
    ids = [
        str(int(v) if pd.notna(v) else v)
        for v in gdf["point_index"].tolist()
    ]
    labels = [f"point {fid}" for fid in ids]
    return write_level(
        out_dir, "sidewalk_segmentized", gdf, ids, labels, value_columns, numeric_columns
    )


def build_qa(root, table, value_columns, numeric_columns, out_dir, simplify_m):
    """Aggregate to quartiers administratifs."""
    qa_path = os.path.join(root, "data", "raw", "quartier_paris_raw.geojson")
    print(f"Building qa from {qa_path} ...")
    qa = gpd.read_file(qa_path)
    qa["c_qu"] = qa["c_qu"].astype(str).str.replace(r"\.0$", "", regex=True)
    scores = aggregate_table(table, "qa_c_qu", value_columns, numeric_columns)
    scores = scores.rename(columns={"qa_c_qu": "c_qu"})
    scores["c_qu"] = scores["c_qu"].astype(str).str.replace(r"\.0$", "", regex=True)
    scores = attach_norm01(scores, numeric_columns)
    merged = qa.merge(scores, on="c_qu", how="inner")
    merged = simplify_polygons(merged, simplify_m)
    ids = merged["c_qu"].astype(str).tolist()
    if "l_qu" in merged.columns:
        labels = [f"QA {name}" for name in merged["l_qu"].tolist()]
    elif "qa_l_qu" in merged.columns:
        labels = [f"QA {name}" for name in merged["qa_l_qu"].tolist()]
    else:
        labels = [f"QA {fid}" for fid in ids]
    return write_level(out_dir, "qa", merged, ids, labels, value_columns, numeric_columns)


def build_pvp(root, table, value_columns, numeric_columns, out_dir, simplify_m):
    """Aggregate to PVP tiles."""
    pvp_path = os.path.join(root, "data", "raw", "pvp_tiles_raw.geojson")
    print(f"Building pvp from {pvp_path} ...")
    pvp = gpd.read_file(pvp_path)
    pvp["numero_pave"] = pvp["numero_pave"].astype(str).str.replace(r"\.0$", "", regex=True)
    scores = aggregate_table(table, "pvp_tile", value_columns, numeric_columns)
    scores = scores.rename(columns={"pvp_tile": "numero_pave"})
    scores["numero_pave"] = scores["numero_pave"].astype(str).str.replace(r"\.0$", "", regex=True)
    scores = attach_norm01(scores, numeric_columns)
    merged = pvp.merge(scores, on="numero_pave", how="inner")
    merged = simplify_polygons(merged, simplify_m)
    ids = merged["numero_pave"].astype(str).tolist()
    labels = [f"PVP {fid}" for fid in ids]
    return write_level(out_dir, "pvp", merged, ids, labels, value_columns, numeric_columns)


def build_arrondissement(root, table, value_columns, numeric_columns, out_dir, simplify_m):
    """Aggregate to arrondissements."""
    ar_path = os.path.join(root, "data", "raw", "arrondissements_raw.geojson")
    print(f"Building arrondissement from {ar_path} ...")
    ar = gpd.read_file(ar_path)
    ar["c_ar"] = ar["c_ar"].astype(str).str.replace(r"\.0$", "", regex=True)

    if "arrondissement_id" in table.columns and table["arrondissement_id"].notna().any():
        key = "arrondissement_id"
    elif "qa_c_ar" in table.columns:
        key = "qa_c_ar"
    else:
        key = "arrondissement"

    scores = aggregate_table(table, key, value_columns, numeric_columns)
    scores = scores.rename(columns={key: "c_ar"})
    scores["c_ar"] = scores["c_ar"].astype(str).str.replace(r"\.0$", "", regex=True)
    scores = attach_norm01(scores, numeric_columns)
    merged = ar.merge(scores, on="c_ar", how="inner")
    merged = simplify_polygons(merged, simplify_m)
    ids = merged["c_ar"].astype(str).tolist()
    if "l_ar" in merged.columns:
        labels = [f"Arrondissement {name}" for name in merged["l_ar"].tolist()]
    else:
        labels = [f"Arrondissement {fid}" for fid in ids]
    return write_level(
        out_dir, "arrondissement", merged, ids, labels, value_columns, numeric_columns
    )


def write_html(root, out_dir, column_order, available_levels, default_level):
    """Fill the MapLibre viewer template and write it next to the GeoJSON layers."""
    template_path = os.path.join(root, "robotability_interactive_map.html")
    html_path = os.path.join(out_dir, "index.html")
    with open(template_path, encoding="utf-8") as handle:
        html = handle.read()
    html = (
        html.replace("__COLUMN_ORDER__", json.dumps(column_order))
        .replace("__AVAILABLE_LEVELS__", json.dumps(available_levels))
        .replace("__DEFAULT_LEVEL__", json.dumps(default_level))
    )
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"Wrote viewer {html_path}")
    return html_path


def selected_levels(args):
    """Expand --level into concrete levels."""
    if "all" in args.level:
        return list(LEVELS)
    return list(dict.fromkeys(args.level))


def main(args):
    """Build GeoJSON layers and a MapLibre HTML viewer for Paris robotability."""
    root = os.path.dirname(os.path.abspath(__file__))
    processed = os.path.join(root, "data", "processed")
    out_dir = args.output_dir or os.path.join(root, "figures", "interactive")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = args.csv_path or os.path.join(processed, "robotability_features_paris.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Run compute_robotability_score.py first."
        )

    table, value_columns, numeric_columns = load_features_table(csv_path)
    levels = selected_levels(args)
    print(f"Levels: {levels}")
    print(f"Numeric columns for 0–1 normalization: {len(numeric_columns)}")

    builders = {
        "sidewalk_full": lambda: build_sidewalk_full(
            root, table, value_columns, numeric_columns, out_dir, args.simplify_m
        ),
        "sidewalk_segmentized": lambda: build_sidewalk_segmentized(
            root, table, value_columns, numeric_columns, out_dir, args.max_points
        ),
        "qa": lambda: build_qa(
            root, table, value_columns, numeric_columns, out_dir, args.simplify_m
        ),
        "pvp": lambda: build_pvp(
            root, table, value_columns, numeric_columns, out_dir, args.simplify_m
        ),
        "arrondissement": lambda: build_arrondissement(
            root, table, value_columns, numeric_columns, out_dir, args.simplify_m
        ),
    }

    built = []
    for level in levels:
        print(f"\n=== {level} ===")
        n = builders[level]()
        if n:
            built.append(level)

    if not built:
        raise RuntimeError("No layers were written.")

    # Keep every level already on disk in the viewer, not only this run.
    available = []
    for level in LEVELS:
        geo = os.path.join(out_dir, f"{level}.geojson")
        attrs = os.path.join(out_dir, f"{level}.attrs.json")
        if os.path.exists(geo) and os.path.exists(attrs):
            available.append(level)
    if not available:
        available = built

    default_level = (
        args.default_level
        if args.default_level in available
        else ("qa" if "qa" in available else available[0])
    )
    html_path = write_html(root, out_dir, value_columns, available, default_level)
    print(
        "\nDone. Open the viewer via a local static server, e.g.\n"
        f"  python -m http.server 8000 --directory {out_dir}\n"
        f"  then visit http://localhost:8000/ ({os.path.basename(html_path)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a MapLibre interactive Paris robotability map "
            "(sidewalk full/segmentized, QA, PVP, arrondissement)."
        )
    )
    parser.add_argument(
        "--level",
        nargs="+",
        default=["all"],
        choices=LEVELS + ["all"],
        help="Aggregation level(s) to build (default: all).",
    )
    parser.add_argument(
        "--default_level",
        default="qa",
        choices=LEVELS,
        help="Initial level shown in the viewer (default: qa).",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="Optional path to robotability_features_paris.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for GeoJSON + index.html "
        "(default: paris_test/figures/interactive).",
    )
    parser.add_argument(
        "--simplify_m",
        type=float,
        default=2.0,
        help="Polygon simplify tolerance in meters (default: 2).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=40000,
        help=(
            "Max segmentized points to embed (default: 40000). "
            "Use 0 for every sample point (large file)."
        ),
    )
    args = parser.parse_args()
    if args.max_points == 0:
        args.max_points = None
    main(args)
