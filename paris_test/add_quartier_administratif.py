import argparse
import gzip
import os
import shutil
from urllib.request import urlretrieve

import geopandas as gpd


def download_geojson(dataset_id, out_path, force=False):
    """Download a Paris Data GeoJSON export and transparently unpack gzip if needed."""
    url = f"https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/geojson"

    if os.path.exists(out_path) and not force:
        print(f"Using existing file: {out_path}")
        return

    print(f"Downloading {dataset_id} ...")
    urlretrieve(url, out_path)

    with open(out_path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        print("Decompressing gzip export ...")
        tmp_path = f"{out_path}.tmp"
        with gzip.open(out_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, out_path)


def main(args):
    """Attach Quartier Administratif identifiers to both Paris sidewalk outputs."""
    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, "data", "raw")
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)

    qa_path = os.path.join(raw_dir, "quartier_paris_raw.geojson")
    sidewalks_path = os.path.join(processed_dir, "sidewalks_paris.geojson")
    sidewalk_widths_path = os.path.join(processed_dir, "sidewalk_widths_paris.geojson")

    download_geojson("quartier_paris", qa_path, force=args.force_download)

    print("Loading layers ...")
    qa = gpd.read_file(qa_path).to_crs("EPSG:2154")
    sidewalks = gpd.read_file(sidewalks_path).to_crs("EPSG:2154")
    sidewalk_widths = gpd.read_file(sidewalk_widths_path).to_crs("EPSG:2154")

    # Keep only QA identifiers and geometry. No need to keep perimeter/surface measurements.
    qa = qa[["n_sq_qu", "c_qu", "c_quinsee", "l_qu", "c_ar", "n_sq_ar", "geometry"]].copy()
    qa = qa.rename(
        columns={
            "n_sq_qu": "qa_n_sq_qu",
            "c_qu": "qa_c_qu",
            "c_quinsee": "qa_c_quinsee",
            "l_qu": "qa_l_qu",
            "c_ar": "qa_c_ar",
            "n_sq_ar": "qa_n_sq_ar",
        }
    )

    qa_cols = ["qa_n_sq_qu", "qa_c_qu", "qa_c_quinsee", "qa_l_qu", "qa_c_ar", "qa_n_sq_ar"]

    # Drop old QA columns if the script is re-run.
    sidewalks = sidewalks.drop(columns=qa_cols, errors="ignore")
    sidewalk_widths = sidewalk_widths.drop(columns=qa_cols, errors="ignore")

    # Use centroids / midpoints so each geometry gets a single QA.
    print("Joining QA to sidewalk polygons ...")
    sidewalks_points = sidewalks.copy()
    sidewalks_points["geometry"] = sidewalks_points.geometry.centroid
    sidewalks_joined = gpd.sjoin(sidewalks_points, qa, how="left", predicate="within").drop(columns=["index_right"])

    print("Joining QA to sidewalk width segments ...")
    sidewalk_widths_points = sidewalk_widths.copy()
    sidewalk_widths_points["geometry"] = sidewalk_widths_points.geometry.interpolate(0.5, normalized=True)
    sidewalk_widths_joined = gpd.sjoin(sidewalk_widths_points, qa, how="left", predicate="within").drop(columns=["index_right"])

    sidewalks = sidewalks.merge(sidewalks_joined[["sidewalk_id"] + qa_cols], on="sidewalk_id", how="left")
    sidewalk_widths = sidewalk_widths.merge(sidewalk_widths_joined[["segment_id"] + qa_cols], on="segment_id", how="left")

    print(f"Unique QA for polygons: {sidewalks['qa_c_qu'].nunique(dropna=True)}")
    print(f"Unique QA for segments: {sidewalk_widths['qa_c_qu'].nunique(dropna=True)}")

    print(f"Writing {sidewalks_path} ...")
    sidewalks.to_file(sidewalks_path, driver="GeoJSON")

    print(f"Writing {sidewalk_widths_path} ...")
    sidewalk_widths.to_file(sidewalk_widths_path, driver="GeoJSON")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach Quartier Administratif identifiers to Paris sidewalk outputs.")
    parser.add_argument("--force_download", action="store_true", help="Redownload quartier_paris even if the file already exists")
    args = parser.parse_args()
    main(args)
