#!/usr/bin/env python3
"""Score robotability features from equirectangular panoramas with a DashScope VLM.

Queries Qwen with the full 360° image, the feature questions from
feature_weights_questions.csv, and compares estimates to metadata.yaml values.

Examples:
    export DASHSCOPE_API_KEY=sk-...
    python sample_nyc_street/vlm_feature_scoring.py --n 5
    python sample_nyc_street/vlm_feature_scoring.py --enable_thinking
    python sample_nyc_street/vlm_feature_scoring.py --results_json sample_nyc_street/vlm_results/vlm_feature_scores.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT_DIR / "feature_weights_questions.csv"
DEFAULT_METADATA = ROOT_DIR / "metadata.yaml"
DEFAULT_IMAGE_DIR = ROOT_DIR / "full"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "vlm_results"

DASHSCOPE_BASE_URLS = {
    "international": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
DEFAULT_MODEL = "qwen3.8-27b"

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert urban robotics analyst. You score sidewalk suitability for a \
small wheeled delivery robot navigating New York City sidewalks.

You are shown one full equirectangular (360°) Google Street View panorama. \
The image wraps left-right around the viewpoint. Score only the {side} sidewalk \
(the sidewalk on the {side} side of the roadway relative to the camera heading). \
Use the whole panorama (front, sides, and behind) when judging each feature.

Each feature is scored on a continuous scale from 0.0 to 1.0 inclusive. \
Follow the polarity given in each question carefully (what 0 and 1 mean differs \
by feature). If something is not visible, give your best estimate from context \
and say so briefly in the optional notes field.

Respond with a single JSON object only (no markdown fences), with this shape:
{{
  "scores": {{
    "<feature_name>": <float between 0 and 1>,
    ...
  }},
  "notes": "<short free-text justification, optional>"
}}

Score every feature listed in the user message. Do not invent extra features.
"""


def system_prompt_for_side(sidewalk_side: str) -> str:
    """Build the system prompt for the left or right sidewalk."""
    side = sidewalk_side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError(f"sidewalk_side must be 'left' or 'right', got {sidewalk_side!r}")
    return SYSTEM_PROMPT_TEMPLATE.format(side=side)


def load_questions(questions_path: Path) -> pd.DataFrame:
    """Load feature questions; keep IGNORED rows for default fill-in."""
    df = pd.read_csv(questions_path)
    required = {"Feature", "Weight", "Question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Questions CSV missing columns: {sorted(missing)}")
    return df


def visual_questions(questions_df: pd.DataFrame) -> pd.DataFrame:
    """Rows that need a VLM estimate (not IGNORED)."""
    mask = questions_df["Question"].astype(str).str.strip().str.upper() != "IGNORED"
    return questions_df.loc[mask].copy()


def ignored_defaults(questions_df: pd.DataFrame) -> dict[str, float]:
    """Feature -> default score for IGNORED rows."""
    defaults = {}
    for _, row in questions_df.iterrows():
        if str(row["Question"]).strip().upper() != "IGNORED":
            continue
        feature = str(row["Feature"])
        raw = row["Default"] if "Default" in questions_df.columns else 1.0
        defaults[feature] = 1.0 if pd.isna(raw) else float(raw)
    return defaults


def format_question_text(question: str, sidewalk_side: str = "left") -> str:
    """Replace XXX with left/right sidewalk side (default: left)."""
    side = sidewalk_side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError(f"sidewalk_side must be 'left' or 'right', got {sidewalk_side!r}")
    text = re.sub(r"\bXXX\b", side, str(question))
    return re.sub(r"\s+", " ", text).strip()


def build_user_prompt(visual_df: pd.DataFrame, sidewalk_side: str = "left") -> str:
    """Build the text part listing features and questions."""
    lines = [
        f"Score the following robotability features for the {sidewalk_side} sidewalk "
        "from the equirectangular panorama.",
        "Return JSON only, as specified in the system prompt.",
        "",
        "Features to score:",
    ]
    for _, row in visual_df.iterrows():
        feature = str(row["Feature"])
        question = format_question_text(row["Question"], sidewalk_side=sidewalk_side)
        lines.append(f"- {feature}: {question}")
    return "\n".join(lines)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode a local image file as base64."""
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def mime_for_path(image_path: Path) -> str:
    """Guess image MIME type from suffix."""
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def extract_json_object(text: str) -> dict:
    """Parse a JSON object from model text, tolerating markdown fences."""
    if not text:
        raise ValueError("Empty model response")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {text[:300]!r}")
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def clamp01(value: float) -> float:
    """Clamp a score to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def normalize_scores(raw_scores: dict, expected_features: list[str]) -> dict[str, float | None]:
    """Keep expected features only and clamp numeric scores to [0, 1]."""
    out: dict[str, float | None] = {}
    for feature in expected_features:
        if feature not in raw_scores:
            out[feature] = None
            continue
        try:
            out[feature] = clamp01(float(raw_scores[feature]))
        except (TypeError, ValueError):
            out[feature] = None
    return out


def make_client(region: str, api_key: str | None, base_url: str | None) -> OpenAI:
    """Create an OpenAI-compatible DashScope client."""
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is not set (pass --api_key or export the env var).")
    url = base_url or DASHSCOPE_BASE_URLS[region]
    return OpenAI(api_key=key, base_url=url)


def query_image_scores(
    client: OpenAI,
    model: str,
    image_path: Path,
    user_prompt: str,
    expected_features: list[str],
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    sidewalk_side: str = "left",
) -> dict:
    """Send one panorama + questions; return parsed scores and raw response meta."""
    image_b64 = encode_image_to_base64(image_path)
    mime = mime_for_path(image_path)
    data_url = f"data:{mime};base64,{image_b64}"

    messages = [
        {"role": "system", "content": system_prompt_for_side(sidewalk_side)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        # Qwen3.8 thinks by default on DashScope; disable unless requested.
        extra_body={"enable_thinking": bool(enable_thinking)},
    )
    elapsed = time.time() - start

    message = response.choices[0].message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    usage = None
    if response.usage is not None:
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)

    parsed = extract_json_object(content)
    raw_scores = parsed.get("scores", parsed)
    if not isinstance(raw_scores, dict):
        raise ValueError(f"Expected scores object, got: {type(raw_scores)}")

    scores = normalize_scores(raw_scores, expected_features)
    system_prompt = system_prompt_for_side(sidewalk_side)
    return {
        "scores": scores,
        "notes": parsed.get("notes"),
        "full_response": content,
        "reasoning_content": reasoning,
        "prompt": {
            "system": system_prompt,
            "user_text": user_prompt,
            "image_path": str(image_path),
            "enable_thinking": bool(enable_thinking),
        },
        "response_time_seconds": round(elapsed, 3),
        "usage": usage,
    }


def load_metadata_images(metadata_path: Path) -> list[dict]:
    """Load image records from metadata.yaml."""
    with open(metadata_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    images = data.get("images") or []
    if not images:
        raise ValueError(f"No images found in {metadata_path}")
    return images


def resolve_image_path(image_dir: Path, record: dict) -> Path:
    """Resolve the equirectangular image path for a metadata record."""
    filename = record.get("original_filename") or record.get("visualization_filename")
    if not filename:
        raise ValueError(f"Metadata record missing filename: {record}")
    path = image_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def merge_scores_with_defaults(
    vlm_scores: dict[str, float | None],
    defaults: dict[str, float],
    all_features: list[str],
) -> dict[str, float | None]:
    """Fill IGNORED features with defaults; keep VLM scores for the rest."""
    merged: dict[str, float | None] = {}
    for feature in all_features:
        if feature in defaults:
            merged[feature] = defaults[feature]
        else:
            merged[feature] = vlm_scores.get(feature)
    return merged


def build_comparison_rows(results: list[dict], visual_features: list[str]) -> pd.DataFrame:
    """Flatten per-image VLM vs metadata feature pairs for plotting."""
    rows = []
    for result in results:
        filename = result["original_filename"]
        gt = result.get("metadata_features") or {}
        pred = result.get("vlm_scores") or {}
        for feature in visual_features:
            pred_val = pred.get(feature)
            gt_val = gt.get(feature)
            if pred_val is None or gt_val is None:
                continue
            rows.append(
                {
                    "original_filename": filename,
                    "feature": feature,
                    "vlm_score": float(pred_val),
                    "metadata_score": float(gt_val),
                    "abs_error": abs(float(pred_val) - float(gt_val)),
                }
            )
    return pd.DataFrame(rows)


def plot_comparisons(comparison_df: pd.DataFrame, output_dir: Path, model_name: str) -> list[Path]:
    """Write scatter and MAE bar charts comparing VLM vs metadata scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if comparison_df.empty:
        print("No overlapping VLM/metadata feature pairs to plot.")
        return written

    features = list(comparison_df["feature"].unique())
    n_features = len(features)
    ncols = min(4, n_features)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for idx, feature in enumerate(features):
        ax = axes[idx // ncols][idx % ncols]
        subset = comparison_df[comparison_df["feature"] == feature]
        ax.scatter(subset["metadata_score"], subset["vlm_score"], alpha=0.75, s=36, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        mae = subset["abs_error"].mean()
        corr = subset["metadata_score"].corr(subset["vlm_score"]) if len(subset) > 1 else np.nan
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("metadata")
        ax.set_ylabel("VLM")
        ax.set_title(f"{feature}\nMAE={mae:.3f}  r={corr:.2f}" if pd.notna(corr) else f"{feature}\nMAE={mae:.3f}")
        ax.set_aspect("equal", adjustable="box")

    for idx in range(n_features, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"VLM vs metadata feature scores ({model_name})", fontsize=12)
    fig.tight_layout()
    scatter_path = output_dir / "vlm_vs_metadata_scatter.png"
    fig.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(scatter_path)

    mae_by_feature = (
        comparison_df.groupby("feature", sort=False)["abs_error"].mean().sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(mae_by_feature))))
    ax.barh(mae_by_feature.index, mae_by_feature.values, color="#4C78A8")
    ax.set_xlabel("Mean absolute error |VLM − metadata|")
    ax.set_title(f"Per-feature MAE ({model_name})")
    ax.set_xlim(0, max(1.0, float(mae_by_feature.max()) * 1.1))
    fig.tight_layout()
    mae_path = output_dir / "vlm_vs_metadata_mae.png"
    fig.savefig(mae_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(mae_path)

    return written


def save_outputs(
    results: list[dict],
    comparison_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> dict[str, Path]:
    """Write JSON results, comparison CSV, and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    results_path = output_dir / "vlm_feature_scores.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": meta, "results": results}, handle, indent=2)
    paths["results_json"] = results_path

    csv_path = output_dir / "vlm_vs_metadata_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    paths["comparison_csv"] = csv_path

    figure_paths = plot_comparisons(comparison_df, output_dir, meta.get("model", DEFAULT_MODEL))
    paths["figures"] = figure_paths
    return paths


def process_images(
    client: OpenAI,
    image_records: list[dict],
    image_dir: Path,
    questions_df: pd.DataFrame,
    model: str,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    rpm: float | None,
    sidewalk_side: str = "left",
) -> list[dict]:
    """Query the VLM for each panorama and attach metadata ground truth."""
    visual_df = visual_questions(questions_df)
    expected_features = visual_df["Feature"].astype(str).tolist()
    all_features = questions_df["Feature"].astype(str).tolist()
    defaults = ignored_defaults(questions_df)
    user_prompt = build_user_prompt(visual_df, sidewalk_side=sidewalk_side)
    system_prompt = system_prompt_for_side(sidewalk_side)
    shared_prompt = {
        "system": system_prompt,
        "user_text": user_prompt,
        "enable_thinking": bool(enable_thinking),
    }

    results = []
    request_timestamps: list[float] = []
    for index, record in enumerate(image_records):
        if rpm and request_timestamps:
            now = time.time()
            request_timestamps = [t for t in request_timestamps if now - t < 60.0]
            if len(request_timestamps) >= rpm:
                wait_time = 60.0 - (now - request_timestamps[0])
                if wait_time > 0:
                    print(f"Rate limit ({rpm} RPM): waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                now = time.time()
                request_timestamps = [t for t in request_timestamps if now - t < 60.0]

        image_path = resolve_image_path(image_dir, record)
        print(f"[{index + 1}/{len(image_records)}] {image_path.name}")

        try:
            response = query_image_scores(
                client=client,
                model=model,
                image_path=image_path,
                user_prompt=user_prompt,
                expected_features=expected_features,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                sidewalk_side=sidewalk_side,
            )
            merged = merge_scores_with_defaults(response["scores"], defaults, all_features)
            result = {
                "original_filename": record["original_filename"],
                "image_path": str(image_path),
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "robotability_score": record.get("robotability_score"),
                "metadata_features": record.get("features") or {},
                "vlm_scores_raw": response["scores"],
                "vlm_scores": merged,
                "notes": response.get("notes"),
                "full_response": response.get("full_response"),
                "reasoning_content": response.get("reasoning_content"),
                "prompt": response.get("prompt"),
                "response_time_seconds": response.get("response_time_seconds"),
                "usage": response.get("usage"),
                "error": None,
            }
            missing = [f for f, v in response["scores"].items() if v is None]
            if missing:
                print(f"  missing/invalid scores: {missing}")
            else:
                print(f"  ok in {response['response_time_seconds']:.1f}s")
        except Exception as exc:  # noqa: BLE001 - keep batch running on API/parse errors
            print(f"  ERROR: {exc}")
            result = {
                "original_filename": record["original_filename"],
                "image_path": str(image_path),
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "robotability_score": record.get("robotability_score"),
                "metadata_features": record.get("features") or {},
                "vlm_scores_raw": {},
                "vlm_scores": merge_scores_with_defaults({}, defaults, all_features),
                "notes": None,
                "full_response": None,
                "reasoning_content": None,
                "prompt": {
                    **shared_prompt,
                    "image_path": str(image_path),
                },
                "response_time_seconds": None,
                "usage": None,
                "error": str(exc),
            }

        results.append(result)
        if rpm:
            request_timestamps.append(time.time())

    return results, shared_prompt


def main(args):
    """Run VLM scoring and/or regenerate comparison plots from saved results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_df = load_questions(Path(args.questions_path))
    visual_features = visual_questions(questions_df)["Feature"].astype(str).tolist()

    if args.results_json:
        with open(args.results_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload["results"]
        meta = payload.get("metadata", {})
        comparison_df = build_comparison_rows(results, visual_features)
        paths = save_outputs(results, comparison_df, output_dir, meta)
        print(f"Wrote comparison CSV to {paths['comparison_csv']}")
        for figure in paths["figures"]:
            print(f"Wrote figure to {figure}")
        return

    image_records = load_metadata_images(Path(args.metadata_path))
    if args.n is not None:
        image_records = image_records[: args.n]
    if args.filenames:
        wanted = {name.strip() for name in args.filenames.split(",") if name.strip()}
        image_records = [r for r in image_records if r["original_filename"] in wanted]
        missing = wanted - {r["original_filename"] for r in image_records}
        for name in sorted(missing):
            print(f"WARNING: filename not in metadata: {name}")

    if not image_records:
        raise SystemExit("No images selected.")

    client = make_client(args.region, args.api_key, args.base_url)
    print(f"Model: {args.model}")
    print(f"Sidewalk side: {args.sidewalk_side}")
    print(f"Images: {len(image_records)}")
    print(f"Visual features: {len(visual_features)}")
    print(f"Endpoint: {client.base_url}")

    results, shared_prompt = process_images(
        client=client,
        image_records=image_records,
        image_dir=Path(args.image_dir),
        questions_df=questions_df,
        model=args.model,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        rpm=args.rpm,
        sidewalk_side=args.sidewalk_side,
    )

    comparison_df = build_comparison_rows(results, visual_features)
    meta = {
        "model": args.model,
        "region": args.region,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_images": len(results),
        "questions_path": str(Path(args.questions_path).resolve()),
        "metadata_path": str(Path(args.metadata_path).resolve()),
        "image_dir": str(Path(args.image_dir).resolve()),
        "enable_thinking": args.enable_thinking,
        "sidewalk_side": args.sidewalk_side,
        "prompt": shared_prompt,
        "visual_features": visual_features,
        "mean_abs_error": float(comparison_df["abs_error"].mean()) if not comparison_df.empty else None,
    }
    paths = save_outputs(results, comparison_df, output_dir, meta)

    print("-" * 60)
    print(f"Results JSON: {paths['results_json']}")
    print(f"Comparison CSV: {paths['comparison_csv']}")
    for figure in paths["figures"]:
        print(f"Figure: {figure}")
    if meta["mean_abs_error"] is not None:
        print(f"Overall MAE: {meta['mean_abs_error']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score robotability features with a DashScope VLM and compare to metadata."
    )
    parser.add_argument("--questions_path", default=str(DEFAULT_QUESTIONS), help="CSV with Feature/Weight/Question/Default.")
    parser.add_argument("--metadata_path", default=str(DEFAULT_METADATA), help="metadata.yaml with ground-truth features.")
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR), help="Directory of full equirectangular JPG images.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/CSV/figures.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope model id (default: qwen3.8-27b).")
    parser.add_argument("--region", default="international", choices=list(DASHSCOPE_BASE_URLS.keys()))
    parser.add_argument("--base_url", default=None, help="Override API base URL.")
    parser.add_argument("--api_key", default=None, help="Override DASHSCOPE_API_KEY.")
    parser.add_argument("--n", type=int, default=None, help="Process only the first N images from metadata.")
    parser.add_argument("--filenames", default=None, help="Comma-separated original_filename values to process.")
    parser.add_argument(
        "--sidewalk_side",
        default="left",
        choices=["left", "right"],
        help="Replace XXX in questions with left or right (default: left).",
    )
    parser.add_argument("--enable_thinking", action="store_true", help="Enable DashScope thinking mode.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=float, default=None, help="Optional requests-per-minute limit.")
    parser.add_argument(
        "--results_json",
        default=None,
        help="Skip API calls and only rebuild comparison plots from a prior vlm_feature_scores.json.",
    )
    args = parser.parse_args()
    main(args)
