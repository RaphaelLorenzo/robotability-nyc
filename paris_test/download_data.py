import argparse
import gzip
import os
import shutil
from urllib.request import urlretrieve


def download_geojson(dataset_id, out_path, force=False, catalog_url=None):
    """Download one Explore v2 GeoJSON export and unpack gzip responses."""
    if catalog_url is None:
        catalog_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
    url = f"{catalog_url}/{dataset_id}/exports/geojson"

    if os.path.exists(out_path) and not force:
        print(f"Skipping {dataset_id}, file already exists: {out_path}")
        return

    print(f"Downloading {dataset_id}")
    print(f"URL: {url}")
    urlretrieve(url, out_path)

    with open(out_path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        print(f"Decompressing gzipped export for {dataset_id}")
        tmp_path = f"{out_path}.tmp"
        with gzip.open(out_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, out_path)

    print(f"Saved to {out_path}\n")


def main(args):
    """Download the Paris open datasets used by the sidewalk testbed."""
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
        "quartiers": {
            "dataset_id": "quartier_paris",
            "path": os.path.join(raw_dir, "quartier_paris_raw.geojson"),
        },
        "lieux_municipaux": {
            "dataset_id": "lieux-municipaux",
            "path": os.path.join(raw_dir, "lieux_municipaux_raw.geojson"),
        },
        "colleges": {
            "dataset_id": "etablissements-scolaires-colleges",
            "path": os.path.join(raw_dir, "colleges_raw.geojson"),
        },
        "ecoles_elementaires": {
            "dataset_id": "etablissements-scolaires-ecoles-elementaires",
            "path": os.path.join(raw_dir, "ecoles_elementaires_raw.geojson"),
        },
        "ecoles_maternelles": {
            "dataset_id": "etablissements-scolaires-maternelles",
            "path": os.path.join(raw_dir, "ecoles_maternelles_raw.geojson"),
        },
        "kiosques_de_presse": {
            "dataset_id": "kiosques-de-presse",
            "path": os.path.join(raw_dir, "kiosques_de_presse_raw.geojson"),
        },
        "terrasses_autorisations": {
            "dataset_id": "terrasses-autorisations",
            "path": os.path.join(raw_dir, "terrasses_autorisations_raw.geojson"),
        },
        "activites": {
            "dataset_id": "que-faire-a-paris-",
            "path": os.path.join(raw_dir, "activites_raw.geojson"),
        },
        "points_arrets_bus": {
            "dataset_id": "plan-de-voirie-mobiliers-urbains-abris-voyageurs-points-darrets-bus",
            "path": os.path.join(raw_dir, "points_arrets_bus_raw.geojson"),
        },
        "sites_touristiques": {
            "dataset_id": "principaux-sites-touristiques-en-ile-de-france0",
            "path": os.path.join(raw_dir, "sites_touristiques_raw.geojson"),
            "catalog_url": "https://data.iledefrance.fr/api/explore/v2.1/catalog/datasets",
        },
        "zones_touristiques_internationales": {
            "dataset_id": "zones-touristiques-internationales",
            "path": os.path.join(raw_dir, "zones_touristiques_internationales_raw.geojson"),
        },
    }

    # Download each dataset as a GeoJSON export.
    for name, dataset in datasets.items():
        print(f"Preparing {name}")
        download_geojson(
            dataset["dataset_id"],
            dataset["path"],
            force=args.force,
            catalog_url=dataset.get("catalog_url"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Paris test datasets from Paris Data.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    args = parser.parse_args()
    main(args)
