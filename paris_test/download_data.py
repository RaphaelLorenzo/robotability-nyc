import argparse
import gzip
import os
import shutil
from urllib.request import urlretrieve


def main(args):
    """Download the three base Paris layers used in this test."""
    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    datasets = {
        "sidewalks": {
            "dataset_id": "plan-de-voirie-trottoirs-emprises",
            "path": os.path.join(raw_dir, "sidewalks_raw.geojson"),
        },
        "pvp_tiles": {
            "dataset_id": "plan-de-voirie-paves-mosaiques-du-plan-de-voirie-de-paris",
            "path": os.path.join(raw_dir, "pvp_tiles_raw.geojson"),
        },
        "arrondissements": {
            "dataset_id": "arrondissements",
            "path": os.path.join(raw_dir, "arrondissements_raw.geojson"),
        },
    }

    base_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"

    # Download each dataset as a GeoJSON export.
    for name, dataset in datasets.items():
        dataset_id = dataset["dataset_id"]
        out_path = dataset["path"]
        url = f"{base_url}/{dataset_id}/exports/geojson"

        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {name}, file already exists: {out_path}")
            continue

        print(f"Downloading {name} from {dataset_id}")
        print(f"URL: {url}")
        urlretrieve(url, out_path)

        # Some Paris Data exports come back gzipped even on the geojson endpoint.
        with open(out_path, "rb") as f:
            magic = f.read(2)
        if magic == b"\x1f\x8b":
            print(f"Decompressing gzipped export for {name}")
            tmp_path = f"{out_path}.tmp"
            with gzip.open(out_path, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp_path, out_path)

        print(f"Saved to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Paris test datasets from Paris Data.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    args = parser.parse_args()
    main(args)
