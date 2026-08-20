import argparse
import io
import os
import random
import re
from pathlib import Path

import pandas as pd
import requests
import yaml
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
repo_root = root.parent
DEFAULT_CSV_PATH = root / "data" / "processed" / "robotability_features_paris.csv"
DEFAULT_OUTPUT_DIR = root / "sample_paris_street"
NOTES_PATH = repo_root / "NOTES.md"
WEIGHTS_PATH = repo_root / "survey_processing" / "feature_weights.csv"
DEFAULT_HEADINGS = {
    "front": 0,
    "right": 90,
    "back": 180,
    "left": 270,
}
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

# Primary raw/absolute columns to show next to each normalized feature score.
FEATURE_RAW_DISPLAY = {
    "sidewalk_width": [("width_m", "{:.2f} m")],
    "pedestrian_density": [("pedestrian_density_base_density_raw", "{:.0f} pop/km2")],
    "crowd_dynamics": [("crowd_dynamics_tourist_sites_count_raw", "{:.0f} tourist sites")],
    "surface_condition": [("surface_condition_anomaly_count_raw", "{:.0f} anomalies")],
    "street_furniture_density": [("__street_furniture_total__", "{:.0f} items")],
    "intersection_safety": [("intersection_safety_accident_rate_raw", "{:.1f} acc/arr")],
    "curb_ramp_availability": [
        ("curb_ramp_availability_escalier_flag", "escalier={:.0f}"),
        ("curb_ramp_availability_accessibilite_flag", "access={:.0f}"),
    ],
    "vehicle_traffic": [("vehicle_traffic_occupation_raw", "{:.3f} occupation")],
    "slope_gradient": [
        ("slope_gradient_mean_raw", "{:.4f} slope"),
        ("slope_gradient_elevation_m", "{:.1f} m elev"),
    ],
    "traffic_management": [("traffic_management_feux_count_raw", "{:.0f} signals")],
    "bicycle_traffic": [("bicycle_traffic_velib_count_raw", "{:.0f} velib")],
    "charging_station_proximity": [
        ("charging_station_proximity_count_raw", "{:.0f} stations")
    ],
}
STREET_FURNITURE_COUNT_COLUMNS = [
    "street_furniture_jardinieres_bancs_corbeilles_count_raw",
    "street_furniture_bornes_barrieres_potelets_count_raw",
    "street_furniture_kiosques_toilettes_panneaux_count_raw",
    "street_furniture_composteurs_count_raw",
    "street_furniture_trilib_count_raw",
    "street_furniture_fontaines_count_raw",
    "street_furniture_anomaly_count_raw",
]


def load_google_maps_api_key():
    """Load the Google Maps API key from env first, then NOTES.md."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if api_key:
        return api_key.strip()

    if not NOTES_PATH.exists():
        raise FileNotFoundError(
            "GOOGLE_MAPS_API_KEY is not set and NOTES.md was not found."
        )

    text = NOTES_PATH.read_text(encoding="utf-8")
    match = re.search(r"GOOGLE_MAPS_API_KEY=([^\s]+)", text)
    if match:
        return match.group(1).strip()

    raise RuntimeError("Could not find GOOGLE_MAPS_API_KEY in env or NOTES.md.")


def load_features(csv_path):
    """Load the Paris robotability feature CSV and validate core columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run compute_robotability_score.py first."
        )

    features = pd.read_csv(csv_path)
    required_columns = {"point_index", "segment_index", "lon", "lat"}
    missing = sorted(required_columns - set(features.columns))
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    features = features.dropna(subset=["lon", "lat"]).copy()
    features["point_index"] = pd.to_numeric(features["point_index"], errors="coerce")
    features["segment_index"] = pd.to_numeric(features["segment_index"], errors="coerce")
    return features


def select_samples(features, args):
    """Select rows to download, either by explicit point ids or a random sample."""
    if args.point_indices:
        selected = features[features["point_index"].isin(args.point_indices)].copy()
        missing = sorted(set(args.point_indices) - set(selected["point_index"].tolist()))
        if missing:
            raise ValueError(f"Requested point_index values not found: {missing}")
        return selected.sort_values("point_index").reset_index(drop=True)

    if len(features) == 0:
        return features.copy()

    sample_n = min(args.sample_n, len(features))
    rng = random.Random(args.random_seed)
    chosen = rng.sample(features.index.tolist(), sample_n)
    selected = features.loc[chosen].copy()
    return selected.sort_values("point_index").reset_index(drop=True)


def fetch_street_view_metadata(session, api_key, lat, lon, radius_m, source):
    """Resolve the nearest Street View panorama and its snapped location."""
    response = session.get(
        "https://maps.googleapis.com/maps/api/streetview/metadata",
        params={
            "location": f"{lat},{lon}",
            "key": api_key,
            "radius": radius_m,
            "source": source,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        error_message = payload.get("error_message")
        detail = f" status={payload.get('status')}"
        if error_message:
            detail = f"{detail} error_message={error_message}"
        raise RuntimeError(f"Street View metadata failed:{detail}")
    return payload


def fetch_street_view_image(
    session,
    api_key,
    lat,
    lon,
    heading,
    fov,
    pitch,
    width,
    height,
    source,
):
    """Download one Street View image for a specific heading."""
    response = session.get(
        "https://maps.googleapis.com/maps/api/streetview",
        params={
            "location": f"{lat},{lon}",
            "size": f"{width}x{height}",
            "heading": round(heading, 2),
            "fov": fov,
            "pitch": pitch,
            "source": source,
            "key": api_key,
        },
        timeout=60,
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def ensure_output_dirs(output_dir):
    """Create the NYC-style output folders: splits, full, full_with_vis."""
    output_dir = Path(output_dir)
    subdirs = {}
    for name in ["splits", "full", "full_with_vis"]:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def format_scalar(value):
    """Format scalars for overlays and JSON-friendly metadata."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def street_furniture_total(row):
    """Sum street-furniture count columns when available."""
    total = 0.0
    found = False
    for column in STREET_FURNITURE_COUNT_COLUMNS:
        if column in row and not pd.isna(row[column]):
            total += float(row[column])
            found = True
    if not found:
        return None
    return total


def format_raw_display_value(row, column, fmt):
    """Format one raw column (or derived total) for overlay display."""
    if column == "__street_furniture_total__":
        value = street_furniture_total(row)
    elif column in row and not pd.isna(row[column]):
        value = float(row[column])
    else:
        return None
    return fmt.format(value)


def build_feature_raw_values(row):
    """Collect interpretable raw/absolute values for each feature."""
    raw_values = {}
    for feature_name, specs in FEATURE_RAW_DISPLAY.items():
        parts = []
        for column, fmt in specs:
            formatted = format_raw_display_value(row, column, fmt)
            if formatted is not None:
                parts.append(formatted)
        if parts:
            raw_values[feature_name] = ", ".join(parts)
    return raw_values


def format_feature_overlay_line(feature_name, score, raw_values):
    """Format one feature line with score and optional absolute values."""
    line = f"{feature_name}: {score:.4f}"
    raw_text = raw_values.get(feature_name)
    if raw_text:
        line = f"{line} ({raw_text})"
    return line


def build_base_name(lat, lon, street_view_metadata, heading_degrees):
    """Build an NYC-style basename: lat,lon_date_pano_d{heading}_z2."""
    date = street_view_metadata.get("date") or "unknown-date"
    pano_id = street_view_metadata.get("pano_id") or "unknown-pano"
    # Keep pano ids filesystem-safe while preserving NYC-like readability.
    pano_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(pano_id))
    heading_int = int(round(float(heading_degrees))) % 360
    return f"{lat:.8f},{lon:.8f}_{date}_{pano_id}_d{heading_int}_z2"


def create_contact_sheet(images_by_direction):
    """Concatenate the four directional images horizontally: front|right|back|left."""
    ordered = [
        images_by_direction["front"],
        images_by_direction["right"],
        images_by_direction["back"],
        images_by_direction["left"],
    ]
    width, height = ordered[0].size
    sheet = Image.new("RGB", (width * len(ordered), height), color=(0, 0, 0))
    for index, image in enumerate(ordered):
        sheet.paste(image, (index * width, 0))
    return sheet


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


def draw_overlay(image, metadata_record):
    """Append a readable metadata panel below the stitched Street View strip."""
    title_font = load_font(28)
    body_font = load_font(22)

    summary_lines = [
        f"lat: {metadata_record['latitude']:.8f}",
        f"lon: {metadata_record['longitude']:.8f}",
        f"sidewalk_id: {metadata_record['closest_sidewalk_id']}",
        f"point_index: {metadata_record['closest_feature_point_index']}",
        f"arrondissement: {metadata_record.get('arrondissement')}",
        f"score: {metadata_record['robotability_score']:.4f}",
    ]
    if metadata_record.get("robotability_score_01") is not None:
        summary_lines.append(f"score_01: {metadata_record['robotability_score_01']:.4f}")
    if metadata_record.get("width_m") is not None:
        summary_lines.append(f"width: {metadata_record['width_m']:.2f} m")

    raw_values = metadata_record.get("feature_raw") or {}
    feature_lines = [
        format_feature_overlay_line(name, value, raw_values)
        for name, value in metadata_record["features"].items()
    ]

    # Two feature columns so long absolute-value lines stay readable.
    mid = (len(feature_lines) + 1) // 2
    feature_columns = [feature_lines[:mid], feature_lines[mid:]]
    line_height = 28
    padding = 24
    title_gap = 10
    column_gap = 40

    panel_rows = 1 + max(len(summary_lines), max((len(col) for col in feature_columns), default=0) + 1)
    panel_height = padding * 2 + title_gap + line_height * panel_rows

    out = Image.new("RGB", (image.width, image.height + panel_height), color=(248, 248, 248))
    out.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)

    y0 = image.height + padding
    draw.text((padding, y0), "Robotability Metadata", font=title_font, fill=(20, 20, 20))
    summary_x = padding
    summary_y = y0 + line_height + title_gap
    for line in summary_lines:
        draw.text((summary_x, summary_y), line, font=body_font, fill=(20, 20, 20))
        summary_y += line_height

    features_x = max(summary_x + 420, image.width // 4)
    draw.text((features_x, y0), "Features", font=title_font, fill=(20, 20, 20))
    for column_index, column_lines in enumerate(feature_columns):
        text_x = features_x + column_index * ((image.width - features_x - padding) // 2 + column_gap // 2)
        text_y = y0 + line_height + title_gap
        for line in column_lines:
            draw.text((text_x, text_y), line, font=body_font, fill=(20, 20, 20))
            text_y += line_height

    return out


def build_metadata_record(row, street_view_metadata, heading_map, base_name, output_paths):
    """Build one NYC-compatible metadata record with all CSV columns."""
    features = {}
    for column in FEATURE_COLUMNS:
        if column in row and not pd.isna(row[column]):
            features[column] = float(row[column])
    feature_raw = build_feature_raw_values(row)

    record = {
        "original_filename": f"{base_name}.jpg",
        "latitude": float(row["lat"]),
        "longitude": float(row["lon"]),
        "heading_degrees": float(heading_map["front"]),
        "closest_sidewalk_id": int(row["segment_index"]) if not pd.isna(row["segment_index"]) else None,
        "closest_feature_point_index": int(row["point_index"]) if not pd.isna(row["point_index"]) else None,
        "closest_distance_meters": 0.0,
        "robotability_score": float(row["robotability_score"]) if "robotability_score" in row and not pd.isna(row["robotability_score"]) else None,
        "robotability_score_01": float(row["robotability_score_01"]) if "robotability_score_01" in row and not pd.isna(row["robotability_score_01"]) else None,
        "width_m": float(row["width_m"]) if "width_m" in row and not pd.isna(row["width_m"]) else None,
        "features": features,
        "feature_raw": feature_raw,
        "visualization_filename": f"{base_name}.jpg",
        "reprojected_filenames": {
            "front": f"{base_name}_front.jpg",
            "back": f"{base_name}_back.jpg",
            "left": f"{base_name}_left.jpg",
            "right": f"{base_name}_right.jpg",
        },
        "arrondissement": format_scalar(row["arrondissement"]) if "arrondissement" in row else None,
        "qa_l_qu": format_scalar(row["qa_l_qu"]) if "qa_l_qu" in row else None,
        "street_view_status": street_view_metadata.get("status"),
        "street_view_pano_id": street_view_metadata.get("pano_id"),
        "street_view_date": street_view_metadata.get("date"),
        "street_view_copyright": street_view_metadata.get("copyright"),
        "street_view_snapped_lat": format_scalar(
            street_view_metadata.get("location", {}).get("lat")
        ),
        "street_view_snapped_lon": format_scalar(
            street_view_metadata.get("location", {}).get("lng")
        ),
        "front_heading": round(heading_map["front"], 2),
        "right_heading": round(heading_map["right"], 2),
        "back_heading": round(heading_map["back"], 2),
        "left_heading": round(heading_map["left"], 2),
        "full_image_path": str(output_paths["full"]),
        "full_with_vis_image_path": (
            str(output_paths["full_with_vis"]) if output_paths.get("full_with_vis") else None
        ),
        "split_image_paths": {
            side: str(output_paths[side]) for side in ["front", "right", "back", "left"]
        },
        "all_columns": {
            column: format_scalar(value) for column, value in row.items()
        },
    }
    return record


def load_existing_metadata(metadata_path):
    """Load an existing metadata.yaml, or raise if resume is requested without it."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: missing {metadata_path}. Run without --resume first."
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    metadata.setdefault("images", [])
    metadata.setdefault("failed", [])
    return metadata


def selected_from_failed_metadata(failed_records):
    """Rebuild sample rows from failed metadata without reloading the CSV."""
    rows = []
    for failed in failed_records:
        columns = failed.get("all_columns")
        if not columns:
            raise ValueError(
                f"Failed record for point_index={failed.get('point_index')} has no all_columns; "
                "cannot resume without reloading the CSV."
            )
        rows.append(columns)
    return pd.DataFrame(rows)


def refresh_record_display_fields(record):
    """Refresh overlay display fields from stored all_columns when possible."""
    columns = record.get("all_columns") or {}
    if not columns:
        return record

    row = pd.Series(columns)
    if "features" not in record or not record["features"]:
        features = {}
        for column in FEATURE_COLUMNS:
            if column in row and not pd.isna(row[column]):
                features[column] = float(row[column])
        record["features"] = features

    record["feature_raw"] = build_feature_raw_values(row)
    if "robotability_score_01" not in record and "robotability_score_01" in row and not pd.isna(row["robotability_score_01"]):
        record["robotability_score_01"] = float(row["robotability_score_01"])
    if "width_m" not in record and "width_m" in row and not pd.isna(row["width_m"]):
        record["width_m"] = float(row["width_m"])
    return record


def regenerate_visualization(record, output_dirs):
    """Recreate full_with_vis from an existing full image and metadata record."""
    record = refresh_record_display_fields(record)
    full_path = Path(record.get("full_image_path") or "")
    if not full_path.exists():
        filename = record.get("original_filename") or record.get("visualization_filename")
        if filename:
            full_path = output_dirs["full"] / filename
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full image for visualization refresh: {full_path}")

    vis_name = record.get("visualization_filename") or record.get("original_filename") or full_path.name
    vis_path = output_dirs["full_with_vis"] / vis_name
    contact_sheet = Image.open(full_path).convert("RGB")
    overlay_image = draw_overlay(contact_sheet, record)
    overlay_image.save(vis_path, quality=95)
    record["full_image_path"] = str(full_path.resolve())
    record["full_with_vis_image_path"] = str(vis_path.resolve())
    return record


def sanitize_error_message(message):
    """Strip API keys from error strings before writing metadata."""
    return re.sub(r"(key=)[^&\s]+", r"\1REDACTED", str(message))


def process_one_sample(row, session, api_key, args, output_dirs):
    """Download one sample's Street View images and write full / full_with_vis outputs."""
    street_view_metadata = fetch_street_view_metadata(
        session=session,
        api_key=api_key,
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        radius_m=args.radius_m,
        source=args.source,
    )
    snapped_lat = float(street_view_metadata["location"]["lat"])
    snapped_lon = float(street_view_metadata["location"]["lng"])
    base_heading = 0.0
    heading_map = {
        direction: (base_heading + offset) % 360.0
        for direction, offset in DEFAULT_HEADINGS.items()
    }
    base_name = build_base_name(
        lat=snapped_lat,
        lon=snapped_lon,
        street_view_metadata=street_view_metadata,
        heading_degrees=heading_map["front"],
    )

    images_by_direction = {}
    output_paths = {}
    for direction in ["front", "right", "back", "left"]:
        image = fetch_street_view_image(
            session=session,
            api_key=api_key,
            lat=snapped_lat,
            lon=snapped_lon,
            heading=heading_map[direction],
            fov=args.fov,
            pitch=args.pitch,
            width=args.width,
            height=args.height,
            source=args.source,
        )
        image_path = output_dirs["splits"] / f"{base_name}_{direction}.jpg"
        image.save(image_path, quality=95)
        images_by_direction[direction] = image
        output_paths[direction] = image_path

    contact_sheet = create_contact_sheet(images_by_direction)
    full_path = output_dirs["full"] / f"{base_name}.jpg"
    contact_sheet.save(full_path, quality=95)
    output_paths["full"] = full_path

    full_with_vis_path = output_dirs["full_with_vis"] / f"{base_name}.jpg"
    output_paths["full_with_vis"] = full_with_vis_path

    record = build_metadata_record(
        row=row,
        street_view_metadata=street_view_metadata,
        heading_map=heading_map,
        base_name=base_name,
        output_paths=output_paths,
    )
    overlay_image = draw_overlay(contact_sheet, record)
    overlay_image.save(full_with_vis_path, quality=95)
    record["full_with_vis_image_path"] = str(full_with_vis_path)
    return record, base_name


def write_metadata_outputs(args, output_dirs, image_records, failed_records):
    """Write metadata.yaml and a flat CSV manifest for successful samples."""
    metadata = {
        "source_image_dir": str(output_dirs["full"].resolve()),
        "visualization_dir": str(output_dirs["full_with_vis"].resolve()),
        "splits_dir": str(output_dirs["splits"].resolve()),
        "feature_csv": str(Path(args.csv_path).resolve()),
        "feature_weights_csv": str(WEIGHTS_PATH.resolve()) if WEIGHTS_PATH.exists() else None,
        "images": image_records,
        "failed": failed_records,
    }
    metadata_path = Path(args.output_dir) / "metadata.yaml"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=False)

    manifest_csv_path = Path(args.output_dir) / "metadata_manifest.csv"
    flat_rows = []
    for record in image_records:
        flat = dict(record.get("all_columns") or {})
        flat.update(
            {
                "original_filename": record["original_filename"],
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "heading_degrees": record["heading_degrees"],
                "street_view_pano_id": record.get("street_view_pano_id"),
                "street_view_date": record.get("street_view_date"),
                "full_image_path": record.get("full_image_path"),
                "full_with_vis_image_path": record.get("full_with_vis_image_path"),
            }
        )
        for side, path in (record.get("split_image_paths") or {}).items():
            flat[f"{side}_image_path"] = path
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(manifest_csv_path, index=False)
    return metadata_path, manifest_csv_path


def main(args):
    """Sample Paris sidewalk points and download four Street View images per point."""
    api_key = load_google_maps_api_key()
    output_dirs = ensure_output_dirs(args.output_dir)
    metadata_path = Path(args.output_dir) / "metadata.yaml"

    if args.resume:
        existing = load_existing_metadata(metadata_path)
        image_records = list(existing.get("images") or [])
        failed_existing = list(existing.get("failed") or [])

        print(
            f"Resuming from {metadata_path}: "
            f"{len(image_records)} existing successes, {len(failed_existing)} failures"
        )

        refreshed_records = []
        for index, record in enumerate(image_records):
            label = record.get("original_filename") or record.get("visualization_filename") or f"image_{index}"
            try:
                refreshed = regenerate_visualization(record, output_dirs)
                refreshed_records.append(refreshed)
                print(f"  refreshed vis: {label}")
            except Exception as exc:
                print(f"  failed vis refresh for {label}: {sanitize_error_message(exc)}")
                refreshed_records.append(record)
        image_records = refreshed_records

        selected = (
            selected_from_failed_metadata(failed_existing)
            if failed_existing
            else pd.DataFrame()
        )
        if len(selected) == 0:
            metadata_path, manifest_csv_path = write_metadata_outputs(
                args=args,
                output_dirs=output_dirs,
                image_records=image_records,
                failed_records=[],
            )
            print(f"Refreshed visualizations for {len(image_records)} samples")
            print(f"Metadata YAML: {metadata_path}")
            print(f"Manifest CSV: {manifest_csv_path}")
            return
        print(f"Retrying {len(selected)} failed downloads ...")
    else:
        print(f"Loading features from {args.csv_path}")
        features = load_features(args.csv_path)
        print(f"Loaded {len(features)} rows")
        selected = select_samples(features, args)
        image_records = []
        print(f"Selected {len(selected)} points from {len(features)} candidate rows")

    if len(selected) == 0:
        print("No sample points selected.")
        return

    session = requests.Session()
    failed_records = []

    for sample_id, (_, row) in enumerate(selected.iterrows()):
        print(
            f"[{sample_id + 1}/{len(selected)}] point_index={int(row['point_index'])} "
            f"segment_index={int(row['segment_index']) if not pd.isna(row['segment_index']) else 'NA'}"
        )
        try:
            record, base_name = process_one_sample(
                row=row,
                session=session,
                api_key=api_key,
                args=args,
                output_dirs=output_dirs,
            )
            image_records.append(record)
            print(f"  wrote {base_name}")
        except Exception as exc:
            print(f"  failed: {sanitize_error_message(exc)}")
            failed_records.append(
                {
                    "point_index": format_scalar(row["point_index"]),
                    "segment_index": format_scalar(row["segment_index"]),
                    "lat": format_scalar(row["lat"]),
                    "lon": format_scalar(row["lon"]),
                    "error": sanitize_error_message(exc),
                    "all_columns": {
                        column: format_scalar(value) for column, value in row.items()
                    },
                }
            )

    metadata_path, manifest_csv_path = write_metadata_outputs(
        args=args,
        output_dirs=output_dirs,
        image_records=image_records,
        failed_records=failed_records,
    )
    print(f"Wrote {len(image_records)} successful samples")
    print(f"Failed: {len(failed_records)}")
    print(f"Metadata YAML: {metadata_path}")
    print(f"Manifest CSV: {manifest_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample Paris robotability points and download Street View imagery."
    )
    parser.add_argument(
        "--csv_path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to robotability_features_paris.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory mirroring sample_nyc_street layout",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=10,
        help="Number of random points to sample when point_indices is not provided",
    )
    parser.add_argument(
        "--point_indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit point_index values to download",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used for sampling rows",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing metadata.yaml, refresh full_with_vis overlays, and retry failed samples",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Street View image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Street View image height in pixels",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Street View field of view",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Street View camera pitch in degrees",
    )
    parser.add_argument(
        "--radius_m",
        type=int,
        default=50,
        help="Search radius in meters for the nearest panorama",
    )
    parser.add_argument(
        "--source",
        default="outdoor",
        choices=["default", "outdoor"],
        help="Street View source filter",
    )
    args = parser.parse_args()
    main(args)
