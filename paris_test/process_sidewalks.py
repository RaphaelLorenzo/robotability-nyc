import argparse
import os

import geopandas as gpd


def main(args):
    """Clean Paris sidewalks and attach PVP tile and arrondissement labels."""
    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, "data", "raw")
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    sidewalks_path = os.path.join(raw_dir, "sidewalks_raw.geojson")
    pvp_tiles_path = os.path.join(raw_dir, "pvp_tiles_raw.geojson")
    arrondissements_path = os.path.join(raw_dir, "arrondissements_raw.geojson")
    out_path = os.path.join(processed_dir, "sidewalks_paris.geojson")

    print("Loading raw layers")
    sidewalks = gpd.read_file(sidewalks_path)
    pvp_tiles = gpd.read_file(pvp_tiles_path)
    arrondissements = gpd.read_file(arrondissements_path)

    # Use Lambert 93 for all spatial operations.
    proj = "EPSG:2154"
    sidewalks = sidewalks.to_crs(proj)
    pvp_tiles = pvp_tiles.to_crs(proj)
    arrondissements = arrondissements.to_crs(proj)

    # Keep only valid sidewalk polygons and split multipart geometries.
    sidewalks = sidewalks[sidewalks.geometry.notna()].copy()
    sidewalks = sidewalks[~sidewalks.geometry.is_empty].copy()
    sidewalks = sidewalks[sidewalks.geometry.is_valid].copy()
    sidewalks = sidewalks.explode(index_parts=False).reset_index(drop=True)

    # Keep only polygon features after explode.
    sidewalks = sidewalks[sidewalks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Add a simple unique id and a few basic geometry fields.
    sidewalks["sidewalk_id"] = sidewalks.index.astype(int)
    sidewalks["sidewalk_area_m2"] = sidewalks.geometry.area
    sidewalks["sidewalk_perimeter_m"] = sidewalks.geometry.length

    # The sidewalk layer already has num_pave; keep it as the first pvp tile source.
    if "num_pave" in sidewalks.columns:
        sidewalks["pvp_tile"] = sidewalks["num_pave"].astype(str)
    else:
        sidewalks["pvp_tile"] = None

    # Spatial join with the tile layer as a fallback and consistency check.
    pvp_tiles = pvp_tiles[["numero_pave", "geometry"]].copy()
    sidewalks_centroids = sidewalks.copy()
    sidewalks_centroids["geometry"] = sidewalks_centroids.geometry.centroid
    sidewalks_with_tiles = gpd.sjoin(
        sidewalks_centroids,
        pvp_tiles,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    missing_pvp_tile = sidewalks_with_tiles["pvp_tile"].isna() | (sidewalks_with_tiles["pvp_tile"] == "None")
    sidewalks_with_tiles.loc[missing_pvp_tile, "pvp_tile"] = sidewalks_with_tiles.loc[missing_pvp_tile, "numero_pave"]
    sidewalks_with_tiles = sidewalks_with_tiles.drop(columns=["numero_pave"])

    # Join arrondissements using centroids so each sidewalk gets one label.
    arrondissements = arrondissements[["c_ar", "l_ar", "geometry"]].copy()
    sidewalks_with_labels = gpd.sjoin(
        sidewalks_with_tiles,
        arrondissements,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    sidewalks_with_labels = sidewalks_with_labels.rename(
        columns={
            "c_ar": "arrondissement_id",
            "l_ar": "arrondissement",
        }
    )

    # Restore the original sidewalk polygons for output.
    sidewalks_out = sidewalks.copy()
    sidewalks_out = sidewalks_out.drop(columns=["pvp_tile"], errors="ignore")
    label_cols = ["sidewalk_id", "pvp_tile", "arrondissement_id", "arrondissement"]
    sidewalks_out = sidewalks_out.merge(
        sidewalks_with_labels[label_cols],
        on="sidewalk_id",
        how="left",
    )
    sidewalks_out = gpd.GeoDataFrame(sidewalks_out, geometry="geometry", crs=proj)

    # Keep only a compact set of columns for this first test.
    keep_cols = [
        "sidewalk_id",
        "num_pave",
        "pvp_tile",
        "arrondissement_id",
        "arrondissement",
        "sidewalk_area_m2",
        "sidewalk_perimeter_m",
        "geometry",
    ]
    keep_cols = [col for col in keep_cols if col in sidewalks_out.columns]
    sidewalks_out = sidewalks_out[keep_cols].copy()

    print(f"Processed sidewalks: {len(sidewalks_out)}")
    print(f"Unique PVP tiles: {sidewalks_out['pvp_tile'].nunique(dropna=True)}")
    print(f"Unique arrondissements: {sidewalks_out['arrondissement'].nunique(dropna=True)}")
    print(f"Writing {out_path}")
    sidewalks_out.to_file(out_path, driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Paris sidewalks and attach tile and arrondissement labels.")
    args = parser.parse_args()
    main(args)
