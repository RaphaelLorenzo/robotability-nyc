from pathlib import Path
import argparse
import math

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import map_coordinates


ROOT_DIR = Path(__file__).resolve().parent
METADATA_PATH = ROOT_DIR / "metadata.yaml"
SPLITS_DIR = ROOT_DIR / "splits"


def wrap_angle_radians(angle):
    """Wrap an angle to the [-pi, pi] interval."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix_y(angle):
    """Return the 3D rotation matrix around the y axis."""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle):
    """Return the 3D rotation matrix around the z axis."""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return np.array(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def equirectangular_to_perspective(image_array, yaw_deg, pitch_deg, hfov_deg, vfov_deg, out_width, out_height):
    """Project an equirectangular panorama into a perspective camera view."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    x_coords = np.linspace(-math.tan(hfov / 2.0), math.tan(hfov / 2.0), out_width, dtype=np.float64)
    y_coords = np.linspace(math.tan(vfov / 2.0), -math.tan(vfov / 2.0), out_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    directions = np.stack([np.ones_like(grid_x), grid_x, grid_y], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    rotation = rotation_matrix_z(yaw) @ rotation_matrix_y(pitch)
    rotated = directions @ rotation.T

    longitude = np.arctan2(rotated[..., 1], rotated[..., 0])
    latitude = np.arcsin(np.clip(rotated[..., 2], -1.0, 1.0))

    image_height, image_width = image_array.shape[:2]
    sample_x = (wrap_angle_radians(longitude) / (2.0 * math.pi) + 0.5) * (image_width - 1)
    sample_y = (0.5 - latitude / math.pi) * (image_height - 1)

    output = np.empty((out_height, out_width, image_array.shape[2]), dtype=np.uint8)
    for channel_index in range(image_array.shape[2]):
        output[..., channel_index] = map_coordinates(
            image_array[..., channel_index],
            [sample_y, sample_x],
            order=1,
            mode="wrap",
        )

    return output


def load_metadata(metadata_path):
    """Read the panorama metadata YAML file."""
    with open(metadata_path, "r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def save_metadata(metadata, metadata_path):
    """Persist updated panorama metadata YAML content."""
    with open(metadata_path, "w", encoding="utf-8") as output_file:
        yaml.safe_dump(metadata, output_file, sort_keys=False, allow_unicode=False)


def main(args):
    """Generate 4 perspective views for every panorama and update metadata."""
    metadata = load_metadata(Path(args.metadata_path))
    splits_dir = Path(args.output_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    relative_yaws = {
        "front": 0.0,
        "right": 90.0,
        "back": 180.0,
        "left": -90.0,
    }

    source_dir = Path(metadata["source_image_dir"])
    for image_record in metadata["images"]:
        image_path = source_dir / image_record["original_filename"]
        image_array = np.asarray(Image.open(image_path).convert("RGB"))
        heading = float(image_record.get("heading_degrees", 0.0))

        reprojected_filenames = {}
        for view_name, relative_yaw in relative_yaws.items():
            yaw = (heading + relative_yaw) % 360.0
            projected = equirectangular_to_perspective(
                image_array=image_array,
                yaw_deg=yaw,
                pitch_deg=args.pitch_deg,
                hfov_deg=args.hfov_deg,
                vfov_deg=args.vfov_deg,
                out_width=args.output_width,
                out_height=args.output_height,
            )

            output_name = f"{image_path.stem}_{view_name}.jpg"
            output_path = splits_dir / output_name
            Image.fromarray(projected).save(output_path, quality=95)
            reprojected_filenames[view_name] = output_name

        image_record["reprojected_filenames"] = reprojected_filenames

    save_metadata(metadata, Path(args.metadata_path))
    print(f"Wrote split images to {splits_dir}")
    print(f"Updated metadata in {args.metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create front/back/left/right perspective crops from equirectangular panoramas.")
    parser.add_argument("--metadata_path", default=str(METADATA_PATH), help="Metadata YAML created by the overlay script.")
    parser.add_argument("--output_dir", default=str(SPLITS_DIR), help="Directory where reprojected views are written.")
    parser.add_argument("--output_width", type=int, default=900, help="Output width for each perspective view.")
    parser.add_argument("--output_height", type=int, default=600, help="Output height for each perspective view.")
    parser.add_argument("--hfov_deg", type=float, default=90.0, help="Horizontal field of view in degrees.")
    parser.add_argument("--vfov_deg", type=float, default=60.0, help="Vertical field of view in degrees.")
    parser.add_argument("--pitch_deg", type=float, default=0.0, help="Optional pitch offset applied to all views.")
    args = parser.parse_args()
    main(args)
