import argparse
import gzip
import os
import shutil
from urllib.request import urlretrieve


def download_geojson(dataset_id, out_path, force=False, catalog_url=None, export_format="geojson"):
    """Download one Explore v2 export and unpack gzip responses."""
    if catalog_url is None:
        catalog_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
    url = f"{catalog_url}/{dataset_id}/exports/{export_format}"

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
        "chantiers": {
            "dataset_id": "chantiers-a-paris",
            "path": os.path.join(raw_dir, "chantiers_a_paris_raw.geojson"),
        },
        "dans_ma_rue": {
            "dataset_id": "dans-ma-rue",
            "path": os.path.join(raw_dir, "dans_ma_rue_raw.geojson"),
        },
        "aires_pietonnes": {
            "dataset_id": "aires-pietonnes",
            "path": os.path.join(raw_dir, "aires_pietonnes_raw.geojson"),
        },
        "zones_de_rencontre": {
            "dataset_id": "zones-de-rencontre",
            "path": os.path.join(raw_dir, "zones_de_rencontre_raw.geojson"),
        },
        "accidentologie": {
            "dataset_id": "accidentologie0",
            "path": os.path.join(raw_dir, "accidentologie_victimes.csv"),
            "export_format": "csv",
        },
        "street_furniture_jardinieres_bancs_corbeilles": {
            "dataset_id": "plan-de-voirie-mobiliers-urbains-jardinieres-bancs-corbeilles-de-rue",
            "path": os.path.join(raw_dir, "street_furniture_jardinieres_bancs_corbeilles_raw.geojson"),
        },
        "street_furniture_bornes_barrieres_potelets": {
            "dataset_id": "plan-de-voirie-mobiliers-urbains-bornes-barrieres-potelets",
            "path": os.path.join(raw_dir, "street_furniture_bornes_barrieres_potelets_raw.geojson"),
        },
        "street_furniture_kiosques_toilettes_panneaux": {
            "dataset_id": "plan-de-voirie-mobiliers-urbains-kiosques-toilettes-publiques-panneaux-publicita",
            "path": os.path.join(raw_dir, "street_furniture_kiosques_toilettes_panneaux_raw.geojson"),
        },
        "street_furniture_composteurs": {
            "dataset_id": "dechets-menagers-points-dapport-volontaire-composteurs",
            "path": os.path.join(raw_dir, "street_furniture_composteurs_raw.geojson"),
        },
        "street_furniture_trilib": {
            "dataset_id": "dechets-menagers-points-dapport-volontaire-stations-trilib",
            "path": os.path.join(raw_dir, "street_furniture_trilib_raw.geojson"),
        },
        "street_furniture_fontaines": {
            "dataset_id": "fontaines-a-boire",
            "path": os.path.join(raw_dir, "street_furniture_fontaines_raw.geojson"),
        },
        "voies_en_escalier": {
            "dataset_id": "plan-de-voirie-voies-en-escalier",
            "path": os.path.join(raw_dir, "voies_en_escalier_raw.geojson"),
        },
        "quartiers_accessibilite": {
            "dataset_id": "perimetresqaa",
            "path": os.path.join(raw_dir, "quartiers_accessibilite_raw.geojson"),
        },
        "feux_tricolores": {
            "dataset_id": "signalisation-tricolore",
            "path": os.path.join(raw_dir, "feux_tricolores_raw.geojson"),
        },
        "ztl": {
            "dataset_id": "ztl",
            "path": os.path.join(raw_dir, "ztl_raw.geojson"),
        },
        "paris_respire": {
            "dataset_id": "secteurs-paris-respire",
            "path": os.path.join(raw_dir, "paris_respire_raw.geojson"),
        },
        "points_de_nivellement": {
            "dataset_id": "plan-de-voirie-points-de-nivellement-etiquettes",
            "path": os.path.join(raw_dir, "points_de_nivellement_raw.geojson"),
        },
        "eclairage_public": {
            "dataset_id": "eclairage-public",
            "path": os.path.join(raw_dir, "eclairage_public_raw.geojson"),
        },
        "pistes_cyclables": {
            "dataset_id": "plan-de-voirie-pistes-cyclables-et-couloirs-de-bus",
            "path": os.path.join(raw_dir, "pistes_cyclables_raw.geojson"),
        },
        "velib_stations": {
            "dataset_id": "velib-emplacement-des-stations",
            "path": os.path.join(raw_dir, "velib_stations_raw.geojson"),
        },
        "arbres": {
            "dataset_id": "les-arbres",
            "path": os.path.join(raw_dir, "arbres_raw.geojson"),
        },
        "referentiel_comptages_routiers": {
            "dataset_id": "referentiel-comptages-routiers",
            "path": os.path.join(raw_dir, "referentiel_comptages_routiers_raw.geojson"),
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
            export_format=dataset.get("export_format", "geojson"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Paris test datasets from Paris Data.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    args = parser.parse_args()
    main(args)
