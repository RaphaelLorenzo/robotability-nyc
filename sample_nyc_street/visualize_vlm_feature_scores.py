#!/usr/bin/env python3
"""Render per-image visualizations from vlm_feature_scores.json.

Each output image stacks the equirectangular panorama above a panel with:
- VLM vs metadata feature scores
- VLM notes
- the full VLM answer in a smaller font

Examples:
    python sample_nyc_street/visualize_vlm_feature_scores.py
    python sample_nyc_street/visualize_vlm_feature_scores.py --results_json sample_nyc_street/vlm_results/vlm_feature_scores.json --n 3
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_JSON = ROOT_DIR / "vlm_results" / "vlm_feature_scores.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "vlm_results" / "vis"
DEFAULT_IMAGE_DIR = ROOT_DIR / "full"

PANEL_BG = (248, 248, 248)
TEXT_COLOR = (20, 20, 20)
MUTED_COLOR = (80, 80, 80)
ACCENT_COLOR = (40, 90, 150)
ERROR_COLOR = (160, 40, 40)


def load_font(font_size: int) -> ImageFont.ImageFont:
    """Load a readable font with a safe fallback."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def load_results(results_json: Path) -> tuple[dict, list[dict]]:
    """Load metadata and per-image results from the scoring JSON."""
    with open(results_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"No results list found in {results_json}")
    return payload.get("metadata") or {}, results


def resolve_image_path(result: dict, image_dir: Path) -> Path:
    """Prefer the stored image path, then fall back to image_dir/filename."""
    stored = result.get("image_path")
    if stored and Path(stored).is_file():
        return Path(stored)
    filename = result.get("original_filename")
    if not filename:
        raise ValueError(f"Result missing original_filename: {result}")
    path = image_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def format_score(value) -> str:
    """Format a numeric score or placeholder."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap text to fit max_width pixels using the given font."""
    if not text:
        return []
    # Approximate character width from an em-ish sample, then refine with bbox.
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    sample_bbox = font.getbbox(sample)
    avg_char_width = max((sample_bbox[2] - sample_bbox[0]) / len(sample), 1.0)
    rough_chars = max(int(max_width / avg_char_width), 10)

    wrapped_lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        for candidate in textwrap.wrap(
            paragraph,
            width=rough_chars,
            break_long_words=True,
            replace_whitespace=False,
        ) or [""]:
            line = candidate
            while font.getbbox(line)[2] - font.getbbox(line)[0] > max_width and len(line) > 1:
                # Shrink until the line fits.
                cut = max(1, int(len(line) * max_width / max(font.getbbox(line)[2], 1)) - 1)
                if cut >= len(line):
                    cut = len(line) - 1
                wrapped_lines.append(line[:cut].rstrip())
                line = line[cut:].lstrip()
            wrapped_lines.append(line)
    return wrapped_lines


def feature_rows(result: dict, visual_features: list[str] | None) -> list[tuple[str, str, str, str]]:
    """Build (feature, metadata, vlm, abs_error) rows for the comparison table."""
    metadata_features = result.get("metadata_features") or {}
    vlm_scores = result.get("vlm_scores_raw") or result.get("vlm_scores") or {}

    if visual_features:
        names = list(visual_features)
    else:
        names = sorted(set(metadata_features) | set(vlm_scores))

    rows = []
    for name in names:
        meta_val = metadata_features.get(name)
        vlm_val = vlm_scores.get(name)
        abs_err = None
        if meta_val is not None and vlm_val is not None:
            try:
                abs_err = abs(float(vlm_val) - float(meta_val))
            except (TypeError, ValueError):
                abs_err = None
        rows.append((name, format_score(meta_val), format_score(vlm_val), format_score(abs_err)))
    return rows


def format_full_vlm_answer(result: dict) -> str:
    """Combine thinking (when present) and the final VLM response for display."""
    reasoning = result.get("reasoning_content")
    answer = result.get("full_response")
    parts = []
    if reasoning:
        parts.append("=== Thinking ===")
        parts.append(str(reasoning).strip())
    if answer:
        if parts:
            parts.append("")
            parts.append("=== Answer ===")
        parts.append(str(answer).strip())
    if not parts:
        return "(no full response)"
    return "\n".join(parts)


def draw_result_overlay(image_path: Path, result: dict, output_path: Path, visual_features: list[str] | None) -> None:
    """Append a VLM/metadata comparison panel below the panorama."""
    image = Image.open(image_path).convert("RGB")
    title_font = load_font(18)
    body_font = load_font(13)
    small_font = load_font(10)

    padding = 12
    line_height = 18
    small_line_height = 13
    title_gap = 6
    section_gap = 10
    text_width = max(image.width - 2 * padding, 100)

    header_bits = [result.get("original_filename") or image_path.name]
    if result.get("latitude") is not None and result.get("longitude") is not None:
        header_bits.append(f"lat={result['latitude']:.6f}, lon={result['longitude']:.6f}")
    if result.get("robotability_score") is not None:
        header_bits.append(f"metadata robotability={float(result['robotability_score']):.4f}")
    if result.get("error"):
        header_bits.append(f"ERROR: {result['error']}")
    header_line = " | ".join(header_bits)

    score_rows = feature_rows(result, visual_features)
    score_lines = ["feature                          meta     vlm    |err|"]
    score_lines.append("-" * 56)
    for name, meta_s, vlm_s, err_s in score_rows:
        score_lines.append(f"{name:<30} {meta_s:>6}  {vlm_s:>6}  {err_s:>6}")

    notes_text = result.get("notes") or "(no notes)"
    notes_lines = wrap_text(str(notes_text), body_font, text_width)

    full_answer = format_full_vlm_answer(result)
    answer_lines = wrap_text(full_answer, small_font, text_width)

    # Cap extremely long answers so the panel stays usable (thinking can be long).
    max_answer_lines = 80 if result.get("reasoning_content") else 40
    answer_truncated = False
    if len(answer_lines) > max_answer_lines:
        answer_lines = answer_lines[:max_answer_lines]
        answer_truncated = True
        answer_lines.append("… [truncated]")

    answer_title = "Full VLM answer"
    if result.get("reasoning_content"):
        answer_title = "Full VLM answer (thinking + response)"

    panel_height = (
        padding
        + line_height  # header title
        + title_gap
        + line_height  # header details
        + section_gap
        + line_height  # scores title
        + title_gap
        + line_height * len(score_lines)
        + section_gap
        + line_height  # notes title
        + title_gap
        + line_height * max(len(notes_lines), 1)
        + section_gap
        + line_height  # full answer title
        + title_gap
        + small_line_height * max(len(answer_lines), 1)
        + padding
    )

    out = Image.new("RGB", (image.width, image.height + panel_height), color=PANEL_BG)
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)

    y = image.height + padding
    draw.text((padding, y), "VLM vs metadata", font=title_font, fill=TEXT_COLOR)
    y += line_height + title_gap
    draw.text((padding, y), header_line, font=body_font, fill=MUTED_COLOR)
    y += line_height + section_gap

    draw.text((padding, y), "Feature scores", font=title_font, fill=TEXT_COLOR)
    y += line_height + title_gap
    for line in score_lines:
        draw.text((padding, y), line, font=body_font, fill=TEXT_COLOR)
        y += line_height
    y += section_gap

    draw.text((padding, y), "VLM notes", font=title_font, fill=ACCENT_COLOR)
    y += line_height + title_gap
    for line in notes_lines or ["(no notes)"]:
        draw.text((padding, y), line, font=body_font, fill=TEXT_COLOR)
        y += line_height
    y += section_gap

    draw.text((padding, y), answer_title, font=title_font, fill=MUTED_COLOR)
    y += line_height + title_gap
    for line in answer_lines:
        color = ERROR_COLOR if answer_truncated and line.startswith("…") else MUTED_COLOR
        draw.text((padding, y), line, font=small_font, fill=color)
        y += small_line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, quality=95)


def main(args):
    """Render overlay visualizations for each VLM scoring result."""
    meta, results = load_results(Path(args.results_json))
    visual_features = meta.get("visual_features")
    if args.n is not None:
        results = results[: args.n]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    written = 0
    for result in results:
        try:
            image_path = resolve_image_path(result, image_dir)
        except FileNotFoundError as exc:
            print(f"SKIP: {exc}")
            continue
        output_path = output_dir / Path(result["original_filename"]).name
        draw_result_overlay(image_path, result, output_path, visual_features)
        written += 1
        print(f"Wrote {output_path}")

    print(f"Wrote {written} visualization(s) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create panorama overlays comparing VLM and metadata feature scores."
    )
    parser.add_argument(
        "--results_json",
        default=str(DEFAULT_RESULTS_JSON),
        help="Path to vlm_feature_scores.json from vlm_feature_scoring.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for annotated images.",
    )
    parser.add_argument(
        "--image_dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Fallback directory for equirectangular images.",
    )
    parser.add_argument("--n", type=int, default=None, help="Render only the first N results.")
    args = parser.parse_args()
    main(args)
