from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer


ROOT_DIR = Path(__file__).resolve().parent
FULL_DIR = ROOT_DIR / "full"
FULL_WITH_VIS_DIR = ROOT_DIR / "full_with_vis"
METADATA_PATH = ROOT_DIR / "metadata.yaml"
FEATURES_PATH = ROOT_DIR.parent / "feature_processing" / "data" / "processed" / "robotability_features.csv"
WEIGHTS_PATH = ROOT_DIR.parent / "survey_processing" / "feature_weights.csv"

LAT_LON_PATTERN = re.compile(r"^(?P<lat>-?\d+\.\d+),(?P<lon>-?\d+\.\d+)_")
POINT_PATTERN = re.compile(r"POINT \((?P<x>-?\d+\.?\d*) (?P<y>-?\d+\.?\d*)\)")
HEADING_PATTERN = re.compile(r"_d(?P<heading>\d+)_")

FEATURE_COLUMNS = [
    "sidewalk_width",
    "pedestrian_density",
    "street_furniture_density",
    "sidewalk_roughness",
    "surface_condition",
    "communication_infrastructure",
    "slope_gradient",
    "charging_station_proximity",
    "curb_ramp_availability",
    "crowd_dynamics",
    "traffic_management",
    "surveillance_coverage",
    "zoning_laws",
    "bike_lane_availability",
    "gps_signal_strength",
    "bicycle_traffic",
    "vehicle_traffic",
    "digital_map_existence",
]


def parse_image_location(filename):
    """Extract latitude and longitude from the panorama filename."""
    match = LAT_LON_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Could not parse lat/lon from filename: {filename}")

    latitude = float(match.group("lat"))
    longitude = float(match.group("lon"))
    return latitude, longitude


def parse_heading(filename):
    """Extract the panorama heading from the filename when available."""
    match = HEADING_PATTERN.search(filename)
    if match is None:
        return 0.0
    return float(match.group("heading")) % 360.0


def load_image_records(image_dir):
    """Collect image metadata from the panorama directory."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    image_records = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        latitude, longitude = parse_image_location(image_path.name)
        x_coord, y_coord = transformer.transform(longitude, latitude)
        image_records.append(
            {
                "image_path": image_path,
                "original_filename": image_path.name,
                "latitude": latitude,
                "longitude": longitude,
                "heading_degrees": parse_heading(image_path.name),
                "x_coord": float(x_coord),
                "y_coord": float(y_coord),
            }
        )

    if not image_records:
        raise ValueError(f"No JPG images found in {image_dir}")

    return image_records


def load_font(font_size):
    """Load a readable font for image overlays with a safe fallback."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def load_feature_weights(weights_path):
    """Load the per-feature survey weights used to compute weighted contributions."""
    weights_df = pd.read_csv(weights_path)
    return dict(zip(weights_df["Feature"], weights_df["Weight"]))


def stream_nearest_feature_rows(image_records, chunk_size):
    """Stream the features CSV and keep the nearest scored point for each image."""
    use_columns = ["geometry", "point_index", "segment_index", "score"] + FEATURE_COLUMNS
    best_matches = [None] * len(image_records)
    best_distances = np.full(len(image_records), np.inf, dtype=np.float64)
    image_coords = np.array([[record["x_coord"], record["y_coord"]] for record in image_records], dtype=np.float64)

    chunk_iterator = pd.read_csv(FEATURES_PATH, usecols=use_columns, chunksize=chunk_size)
    for chunk in chunk_iterator:
        xy_values = chunk["geometry"].str.extract(POINT_PATTERN).astype(float)
        point_coords = xy_values.to_numpy(dtype=np.float64)

        diff_x = point_coords[:, 0][:, None] - image_coords[None, :, 0]
        diff_y = point_coords[:, 1][:, None] - image_coords[None, :, 1]
        distances = np.sqrt(diff_x * diff_x + diff_y * diff_y)

        chunk_min_distances = distances.min(axis=0)
        chunk_best_indices = distances.argmin(axis=0)

        for image_index, chunk_distance in enumerate(chunk_min_distances):
            if chunk_distance >= best_distances[image_index]:
                continue

            row = chunk.iloc[int(chunk_best_indices[image_index])]
            best_distances[image_index] = float(chunk_distance)
            best_matches[image_index] = row.to_dict()

    if any(match is None for match in best_matches):
        raise RuntimeError("Failed to match at least one image to a feature row")

    return best_matches, best_distances.tolist()


def build_metadata_records(image_records, best_matches, best_distances, feature_weights):
    """Create serializable metadata records for each image."""
    metadata_records = []

    for image_record, match_row, match_distance in zip(image_records, best_matches, best_distances):
        features = {}
        for column in FEATURE_COLUMNS:
            weight = float(feature_weights.get(column, 1.0))
            weighted_value = float(match_row[column])
            features[column] = weighted_value / weight if weight != 0.0 else weighted_value

        base_stem = image_record["image_path"].stem
        metadata_records.append(
            {
                "original_filename": image_record["original_filename"],
                "latitude": image_record["latitude"],
                "longitude": image_record["longitude"],
                "heading_degrees": image_record["heading_degrees"],
                "closest_sidewalk_id": int(match_row["segment_index"]),
                "closest_feature_point_index": int(match_row["point_index"]),
                "closest_distance_meters": float(match_distance),
                "robotability_score": float(match_row["score"]),
                "features": features,
                "visualization_filename": image_record["original_filename"],
                "reprojected_filenames": {
                    "front": f"{base_stem}_front.jpg",
                    "back": f"{base_stem}_back.jpg",
                    "left": f"{base_stem}_left.jpg",
                    "right": f"{base_stem}_right.jpg",
                },
            }
        )

    return metadata_records


def draw_overlay(image_path, metadata_record, output_path):
    """Append a readable metadata panel below the panorama image."""
    image = Image.open(image_path).convert("RGB")
    title_font = load_font(18)
    body_font = load_font(14)

    summary_lines = [
        f"lat: {metadata_record['latitude']:.8f}",
        f"lon: {metadata_record['longitude']:.8f}",
        f"sidewalk_id: {metadata_record['closest_sidewalk_id']}",
        f"point_index: {metadata_record['closest_feature_point_index']}",
        f"distance_m: {metadata_record['closest_distance_meters']:.2f}",
        f"score: {metadata_record['robotability_score']:.4f}",
    ]
    feature_lines = [f"{name}: {value:.4f}" for name, value in metadata_record["features"].items()]

    # Two feature columns so the panel stays compact under a 1024-wide panorama.
    mid = (len(feature_lines) + 1) // 2
    feature_columns = [feature_lines[:mid], feature_lines[mid:]]
    line_height = 20
    padding = 14
    title_gap = 8
    column_gap = 24

    panel_rows = 1 + max(len(summary_lines), max((len(col) for col in feature_columns), default=0) + 1)
    panel_height = padding * 2 + title_gap + line_height * panel_rows

    out = Image.new("RGB", (image.width, image.height + panel_height), color=(248, 248, 248))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)

    y0 = image.height + padding
    draw.text((padding, y0), "Robotability Metadata", font=title_font, fill=(20, 20, 20))
    summary_x = padding
    summary_y = y0 + line_height + title_gap
    for line in summary_lines:
        draw.text((summary_x, summary_y), line, font=body_font, fill=(20, 20, 20))
        summary_y += line_height

    features_x = max(summary_x + 250, image.width // 3)
    draw.text((features_x, y0), "Features", font=title_font, fill=(20, 20, 20))
    feature_area_width = max(image.width - features_x - padding, 1)
    for column_index, column_lines in enumerate(feature_columns):
        text_x = features_x + column_index * (feature_area_width // 2 + column_gap // 2)
        text_y = y0 + line_height + title_gap
        for line in column_lines:
            draw.text((text_x, text_y), line, font=body_font, fill=(20, 20, 20))
            text_y += line_height

    out.save(output_path, quality=95)


def main(args):
    """Build metadata.yaml and annotated panorama visualizations."""
    FULL_WITH_VIS_DIR.mkdir(parents=True, exist_ok=True)

    image_records = load_image_records(Path(args.image_dir))
    feature_weights = load_feature_weights(Path(args.weights_path))
    best_matches, best_distances = stream_nearest_feature_rows(image_records, args.chunk_size)
    metadata_records = build_metadata_records(image_records, best_matches, best_distances, feature_weights)

    for image_record, metadata_record in zip(image_records, metadata_records):
        output_path = FULL_WITH_VIS_DIR / metadata_record["visualization_filename"]
        draw_overlay(image_record["image_path"], metadata_record, output_path)

    metadata = {
        "source_image_dir": str(Path(args.image_dir).resolve()),
        "visualization_dir": str(FULL_WITH_VIS_DIR.resolve()),
        "splits_dir": str((ROOT_DIR / "splits").resolve()),
        "feature_csv": str(FEATURES_PATH.resolve()),
        "feature_weights_csv": str(Path(args.weights_path).resolve()),
        "feature_weights": {key: float(value) for key, value in feature_weights.items() if key in FEATURE_COLUMNS},
        "images": metadata_records,
    }

    with open(args.metadata_path, "w", encoding="utf-8") as output_file:
        yaml.safe_dump(metadata, output_file, sort_keys=False, allow_unicode=False)

    print(f"Wrote metadata to {args.metadata_path}")
    print(f"Wrote {len(metadata_records)} annotated images to {FULL_WITH_VIS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match sample panoramas to robotability features and render overlays.")
    parser.add_argument("--image_dir", default=str(FULL_DIR), help="Directory with full equirectangular JPG images.")
    parser.add_argument("--metadata_path", default=str(METADATA_PATH), help="Output metadata YAML path.")
    parser.add_argument("--weights_path", default=str(WEIGHTS_PATH), help="CSV containing the robotability feature weights.")
    parser.add_argument("--chunk_size", type=int, default=50000, help="CSV chunk size used when streaming features.")
    args = parser.parse_args()
    main(args)
