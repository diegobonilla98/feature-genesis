from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torchvision.transforms import functional as TF

from feature_genesis.activations import ActivationCache
from feature_genesis.attribution.spatial import spatial_feature_evidence
from feature_genesis.config import Config
from feature_genesis.data.factory import DatasetBundle, select_split
from feature_genesis.hashing import stable_seed
from feature_genesis.sae.core import SparseAutoencoder


LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "feature_id",
        "short_label",
        "description",
        "confidence",
        "scope",
        "evidence",
        "ambiguities",
        "alternative_labels",
    ],
    "properties": {
        "feature_id": {"type": "integer"},
        "short_label": {"type": "string"},
        "description": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "scope": {"type": "string", "enum": ["local_visual", "global_visual", "class", "mixed", "unclear"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "alternative_labels": {"type": "array", "items": {"type": "string"}},
    },
}


def build_feature_card(
    config: Config,
    sae: SparseAutoencoder,
    cache: ActivationCache,
    feature_id: int,
    device: torch.device,
) -> dict[str, Any]:
    evidence = spatial_feature_evidence(
        sae,
        cache,
        feature_id,
        device,
        config.data.image_size,
        config.evaluation.part_radius_fraction,
        config.evaluation.top_examples,
        config.sae.batch_size,
    )
    scores = image_feature_scores(
        sae,
        cache,
        feature_id,
        device,
        config.sae.batch_size,
    )
    groups = contrastive_example_groups(config, cache, scores, feature_id)
    labels = np.asarray(cache.labels[cache.image_offsets[:-1]], dtype=np.int64)
    class_summary = class_activation_summary(
        scores,
        labels,
        cache.metadata.get("class_names", []),
    )
    prompt = (
        "Infer a falsifiable visual hypothesis for one sparse autoencoder feature from contrastive evidence. "
        "Distinguish entity, part, appearance, geometry, pose, context, spatial role, and mixed selectivity. "
        "Strong positives, borderline positives, and matched hard negatives are discovery evidence. "
        "Held-out examples are reserved for validation and must not be used to form the initial hypothesis."
    )
    return {
        "feature_id": feature_id,
        "model_step": int(cache.metadata["checkpoint_step"]),
        "cache_split": str(cache.metadata["split"]),
        "layer": cache.metadata["layer"],
        "sae_kind": sae.kind,
        "sae_metadata": sae.metadata(),
        "evidence": evidence,
        "contrastive_groups": groups,
        "score_summary": {
            "active_images": int((scores > 0).sum()),
            "activation_frequency_per_image": float((scores > 0).mean()),
            "maximum": float(scores.max()),
            "median_when_active": float(np.median(scores[scores > 0])) if (scores > 0).any() else 0.0,
            "class_confounding": class_summary,
        },
        "system_prompt": prompt,
        "label_schema": LABEL_SCHEMA,
    }


def build_feature_card_from_scores(
    config: Config,
    sae: SparseAutoencoder,
    cache: ActivationCache,
    feature_id: int,
    scores: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(cache.labels[cache.image_offsets[:-1]], dtype=np.int64)
    groups = contrastive_example_groups(config, cache, scores, feature_id)
    class_summary = class_activation_summary(
        scores,
        labels,
        cache.metadata.get("class_names", []),
    )
    active = scores > 0
    prompt = (
        "Infer a falsifiable visual hypothesis for one sparse autoencoder feature from contrastive evidence. "
        "Distinguish entity, part, appearance, geometry, pose, context, spatial role, and mixed selectivity. "
        "Strong positives, borderline positives, and matched hard negatives are discovery evidence. "
        "Held-out examples are reserved for validation and must not be used to form the initial hypothesis."
    )
    evidence = {
        "feature_id": feature_id,
        "activation_frequency_per_image": float(active.mean()),
        "active_images": int(active.sum()),
        "localized_active_images": 0,
        "attribute_certainty_available": False,
        "part_hit_rate": 0.0,
        "median_nearest_part_distance": None,
        "top_images": [],
        "top_attributes": [],
        "top_parts": [],
    }
    return {
        "feature_id": feature_id,
        "model_step": int(cache.metadata["checkpoint_step"]),
        "cache_split": str(cache.metadata["split"]),
        "layer": cache.metadata["layer"],
        "sae_kind": sae.kind,
        "sae_metadata": sae.metadata(),
        "evidence": evidence,
        "contrastive_groups": groups,
        "score_summary": {
            "active_images": int(active.sum()),
            "activation_frequency_per_image": float(active.mean()),
            "maximum": float(scores.max()),
            "median_when_active": (
                float(np.median(scores[active])) if active.any() else 0.0
            ),
            "class_confounding": class_summary,
        },
        "system_prompt": prompt,
        "label_schema": LABEL_SCHEMA,
    }


def class_activation_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    active = scores > 0
    class_count = max(len(class_names), int(labels.max()) + 1 if len(labels) else 0)
    mass = np.bincount(
        labels,
        weights=np.maximum(scores, 0),
        minlength=class_count,
    ).astype(np.float64)
    probability = mass / max(float(mass.sum()), 1e-12)
    nonzero = probability > 0
    entropy = float(-(probability[nonzero] * np.log(probability[nonzero])).sum())
    effective_classes = float(np.exp(entropy))
    top_count = min(32, len(scores))
    top = np.argsort(scores)[::-1][:top_count]
    top_groups = np.bincount(labels[top], minlength=class_count)
    modal_label = int(np.argmax(top_groups)) if len(top_groups) else -1
    modal_share = float(top_groups[modal_label] / max(top_count, 1)) if modal_label >= 0 else 0.0
    return {
        "active_classes": int(len(np.unique(labels[active]))) if active.any() else 0,
        "effective_classes_by_activation_mass": effective_classes,
        "top_32_modal_class_share": modal_share,
        "top_32_modal_class": class_names[modal_label]
        if 0 <= modal_label < len(class_names)
        else str(modal_label),
        "top_32_unique_classes": int(len(np.unique(labels[top]))) if len(top) else 0,
    }


@torch.no_grad()
def image_feature_scores(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    feature_id: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    image_count = len(cache.image_offsets) - 1
    scores = np.zeros(image_count, dtype=np.float32)
    tokens_per_image = max(1, int(cache.metadata["tokens_per_image"]))
    images_per_group = max(1, batch_size // tokens_per_image)
    for image_start in range(0, image_count, images_per_group):
        image_end = min(image_count, image_start + images_per_group)
        row_start = int(cache.image_offsets[image_start])
        row_end = int(cache.image_offsets[image_end])
        inputs = torch.from_numpy(
            np.array(cache.activations[row_start:row_end], dtype=np.float32, copy=True)
        ).to(device)
        values = sae.encode(inputs, inference=True)[:, feature_id]
        for image_index in range(image_start, image_end):
            local_start = int(cache.image_offsets[image_index]) - row_start
            local_end = int(cache.image_offsets[image_index + 1]) - row_start
            scores[image_index] = float(
                values[local_start:local_end].amax().detach().cpu()
            )
    return scores


def contrastive_example_groups(
    config: Config,
    cache: ActivationCache,
    scores: np.ndarray,
    feature_id: int,
) -> dict[str, list[dict[str, Any]]]:
    image_ids = np.asarray(cache.image_ids[cache.image_offsets[:-1]], dtype=np.int64)
    dataset_indices = np.asarray(cache.dataset_indices[cache.image_offsets[:-1]], dtype=np.int64)
    labels = np.asarray(cache.labels[cache.image_offsets[:-1]], dtype=np.int64)
    class_names = cache.metadata.get("class_names", [])
    active = np.flatnonzero(scores > 0)
    inactive = np.flatnonzero(scores <= 0)
    descending = active[np.argsort(scores[active])[::-1]]
    ascending = active[np.argsort(scores[active])]
    top_count = min(config.evaluation.gemini_top_positives, len(descending))
    borderline_count = min(config.evaluation.gemini_borderline_positives, max(0, len(active) - top_count))
    validation_positive_count = min(
        config.evaluation.gemini_validation_positives,
        max(0, len(active) - top_count - borderline_count),
    )
    top = diverse_class_indices(
        descending,
        labels,
        top_count,
        config.evaluation.gemini_max_examples_per_class,
    )
    used = set(top.tolist())
    borderline = np.asarray(
        diverse_class_indices(
            np.asarray([index for index in ascending if int(index) not in used]),
            labels,
            borderline_count,
            config.evaluation.gemini_max_examples_per_class,
        ),
        dtype=np.int64,
    )
    used.update(borderline.tolist())
    spectrum, spectrum_bands = activation_spectrum_indices(
        active,
        scores,
        labels,
        config.evaluation.gemini_activation_bins,
        config.evaluation.gemini_examples_per_bin,
        config.evaluation.gemini_max_examples_per_class,
        used,
    )
    used.update(spectrum.tolist())
    heldout_descending = np.asarray(
        [index for index in descending if int(index) not in used],
        dtype=np.int64,
    )
    validation_positive = diverse_class_indices(
        heldout_descending,
        labels,
        validation_positive_count,
        config.evaluation.gemini_max_examples_per_class,
    )
    used.update(validation_positive.tolist())
    generator = np.random.default_rng(
        stable_seed(config.project.seed, "feature-card", int(cache.metadata["checkpoint_step"]), feature_id)
    )
    inactive = inactive.copy()
    generator.shuffle(inactive)
    preferred_labels = set(labels[top].tolist())
    matched_inactive = [int(index) for index in inactive if int(labels[index]) in preferred_labels]
    unmatched_inactive = [int(index) for index in inactive if int(labels[index]) not in preferred_labels]
    negative_order = matched_inactive + unmatched_inactive
    hard_count = min(config.evaluation.gemini_hard_negatives, len(negative_order))
    hard_negative = np.asarray(negative_order[:hard_count], dtype=np.int64)
    validation_negative_count = min(
        config.evaluation.gemini_validation_negatives,
        max(0, len(negative_order) - hard_count),
    )
    validation_negative = np.asarray(
        negative_order[hard_count : hard_count + validation_negative_count],
        dtype=np.int64,
    )

    def records(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "image_index": int(index),
                "image_id": int(image_ids[index]),
                "dataset_index": int(dataset_indices[index]),
                "label_id": int(labels[index]),
                "class_name": class_names[int(labels[index])] if int(labels[index]) < len(class_names) else str(labels[index]),
                "activation": float(scores[index]),
            }
            for index in indices
        ]

    output = {
        "top_positive": records(top),
        "borderline_positive": records(borderline),
        "hard_negative": records(hard_negative),
        "validation_positive": records(validation_positive),
        "validation_negative": records(validation_negative),
    }
    output["activation_spectrum"] = records(spectrum)
    for row, band in zip(output["activation_spectrum"], spectrum_bands, strict=True):
        row["activation_band"] = int(band)
    return output


def diverse_class_indices(
    ordered_indices: np.ndarray,
    labels: np.ndarray,
    count: int,
    maximum_per_class: int,
) -> np.ndarray:
    selected = []
    class_counts: dict[int, int] = {}
    for index in ordered_indices:
        label = int(labels[index])
        if class_counts.get(label, 0) >= maximum_per_class:
            continue
        selected.append(int(index))
        class_counts[label] = class_counts.get(label, 0) + 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected_set = set(selected)
        for index in ordered_indices:
            if int(index) in selected_set:
                continue
            selected.append(int(index))
            if len(selected) >= count:
                break
    return np.asarray(selected, dtype=np.int64)


def activation_spectrum_indices(
    active: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int,
    per_bin: int,
    maximum_per_class: int,
    excluded: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    if len(active) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    minimum = float(scores[active].min())
    maximum = float(scores[active].max())
    edges = np.linspace(minimum, maximum, bins + 1)
    selected = []
    bands = []
    class_counts: dict[int, int] = {}
    for band in range(bins - 1, -1, -1):
        lower = edges[band]
        upper = edges[band + 1]
        candidates = active[
            (scores[active] >= lower)
            & ((scores[active] <= upper) if band == bins - 1 else (scores[active] < upper))
        ]
        center = (lower + upper) / 2
        candidates = candidates[np.argsort(np.abs(scores[candidates] - center))]
        band_selected = []
        for index in candidates:
            resolved_index = int(index)
            label = int(labels[resolved_index])
            if resolved_index in excluded:
                continue
            if class_counts.get(label, 0) >= maximum_per_class:
                continue
            band_selected.append(resolved_index)
            class_counts[label] = class_counts.get(label, 0) + 1
            if len(band_selected) >= per_bin:
                break
        selected.extend(band_selected)
        bands.extend([band] * len(band_selected))
    return np.asarray(selected, dtype=np.int64), np.asarray(bands, dtype=np.int64)


def add_blind_validation_tiles(
    config: Config,
    card: dict[str, Any],
) -> dict[str, Any]:
    tiles = []
    for group_name, target in [
        ("validation_positive", 1),
        ("validation_negative", 0),
    ]:
        for example in card["contrastive_groups"][group_name]:
            tiles.append(
                {
                    "target": target,
                    "group": group_name,
                    "example": example,
                }
            )
    generator = np.random.default_rng(
        stable_seed(
            config.project.seed,
            "blind-validation-panel",
            int(card["model_step"]),
            int(card["feature_id"]),
        )
    )
    generator.shuffle(tiles)
    for position, tile in enumerate(tiles, start=1):
        tile["tile_id"] = f"T{position:02d}"
    card["blind_validation_tiles"] = tiles
    return card


def write_feature_card(card: dict[str, Any], root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    feature_id = int(card["feature_id"])
    path = root / f"feature_{feature_id:05d}.json"
    path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return path


def render_feature_contact_sheet(
    config: Config,
    bundle: DatasetBundle,
    card: dict[str, Any],
    root: str | Path,
    columns: int = 4,
) -> Path:
    examples = card["evidence"]["top_images"]
    dataset = select_split(bundle, str(card["cache_split"]))
    tile_size = config.data.image_size
    caption_height = 28
    rows = (len(examples) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for position, example in enumerate(examples):
        sample = dataset[int(example["dataset_index"])]
        image = _to_display_image(sample["image"], config.data.name)
        location = example["location"]
        if location and all(value == value for value in location):
            local_draw = ImageDraw.Draw(image)
            x, y = float(location[0]), float(location[1])
            radius = max(2, tile_size // 32)
            local_draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=max(1, tile_size // 80))
        column = position % columns
        row = position // columns
        x0 = column * tile_size
        y0 = row * (tile_size + caption_height)
        canvas.paste(image, (x0, y0))
        caption = f"id={example['image_id']} a={example['activation']:.3f}"
        draw.text((x0 + 3, y0 + tile_size + 5), caption, fill="black")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    feature_id = int(card["feature_id"])
    path = root / f"feature_{feature_id:05d}.png"
    canvas.save(path)
    return path


@torch.no_grad()
def render_contrastive_contact_sheet(
    config: Config,
    bundle: DatasetBundle,
    cache: ActivationCache,
    sae: SparseAutoencoder,
    card: dict[str, Any],
    root: str | Path,
    group_names: list[str],
    filename_suffix: str,
    device: torch.device,
    columns: int = 2,
) -> Path:
    dataset = select_split(bundle, str(card["cache_split"]))
    tile_size = config.data.image_size
    pair_width = tile_size * 2
    caption_height = 24
    examples = []
    for group_name in group_names:
        for position, example in enumerate(card["contrastive_groups"][group_name], start=1):
            examples.append((group_name, position, example))
    rows = (len(examples) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * pair_width, rows * (tile_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    group_codes = {
        "top_positive": "P",
        "borderline_positive": "B",
        "hard_negative": "N",
        "validation_positive": "V+",
        "validation_negative": "V-",
    }
    for panel_index, (group_name, position, example) in enumerate(examples):
        sample = dataset[int(example["dataset_index"])]
        image = _to_display_image(sample["image"], config.data.name)
        response = feature_response_map(
            sae,
            cache,
            int(example["image_index"]),
            int(card["feature_id"]),
            device,
        )
        overlay = response_overlay(image, response)
        column = panel_index % columns
        row = panel_index // columns
        x0 = column * pair_width
        y0 = row * (tile_size + caption_height)
        canvas.paste(image, (x0, y0))
        canvas.paste(overlay, (x0 + tile_size, y0))
        caption = (
            f"{group_codes[group_name]}{position} "
            f"activation={float(example['activation']):.3f}"
        )
        draw.text((x0 + 3, y0 + tile_size + 4), caption, fill="black")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    feature_id = int(card["feature_id"])
    path = root / f"feature_{feature_id:05d}_{filename_suffix}.png"
    canvas.save(path)
    return path


@torch.no_grad()
def render_blind_validation_contact_sheet(
    config: Config,
    bundle: DatasetBundle,
    cache: ActivationCache,
    sae: SparseAutoencoder,
    card: dict[str, Any],
    root: str | Path,
    device: torch.device,
    columns: int = 2,
) -> Path:
    dataset = select_split(bundle, str(card["cache_split"]))
    tile_size = config.data.image_size
    pair_width = tile_size * 2
    caption_height = 24
    tiles = card["blind_validation_tiles"]
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * pair_width, rows * (tile_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for panel_index, tile in enumerate(tiles):
        example = tile["example"]
        sample = dataset[int(example["dataset_index"])]
        image = _to_display_image(sample["image"], config.data.name)
        response = feature_response_map(
            sae,
            cache,
            int(example["image_index"]),
            int(card["feature_id"]),
            device,
        )
        overlay = response_overlay(image, response)
        column = panel_index % columns
        row = panel_index // columns
        x0 = column * pair_width
        y0 = row * (tile_size + caption_height)
        canvas.paste(image, (x0, y0))
        canvas.paste(overlay, (x0 + tile_size, y0))
        draw.text((x0 + 3, y0 + tile_size + 4), tile["tile_id"], fill="black")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    feature_id = int(card["feature_id"])
    path = root / f"feature_{feature_id:05d}_validation_blind.png"
    canvas.save(path)
    return path


@torch.no_grad()
def render_individual_discovery_evidence(
    config: Config,
    bundle: DatasetBundle,
    cache: ActivationCache,
    sae: SparseAutoencoder,
    card: dict[str, Any],
    root: str | Path,
    device: torch.device,
) -> list[Path]:
    dataset = select_split(bundle, str(card["cache_split"]))
    feature_root = Path(root) / f"feature_{int(card['feature_id']):05d}" / "discovery"
    feature_root.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = []
    groups = [
        ("top_positive", "top"),
        ("activation_spectrum", "spectrum"),
        ("borderline_positive", "weak"),
        ("hard_negative", "negative"),
    ]
    for group_name, code in groups:
        for position, example in enumerate(card["contrastive_groups"][group_name], start=1):
            sample = dataset[int(example["dataset_index"])]
            image = _to_display_image(sample["image"], config.data.name).convert("RGB")
            if group_name == "activation_spectrum":
                prefix = f"{code}_b{int(example['activation_band']):02d}_{position:02d}"
            else:
                prefix = f"{code}_{position:02d}"
            if group_name == "hard_negative":
                path = feature_root / f"{prefix}_original.png"
                image.resize((256, 256), Image.Resampling.NEAREST).save(path)
                paths.append(path)
                manifest.append(
                    {
                        "id": prefix,
                        "group": group_name,
                        "activation": float(example["activation"]),
                        "class_name": example["class_name"],
                        "paths": [str(path)],
                        "region": None,
                    }
                )
                continue
            response = feature_response_map(
                sae,
                cache,
                int(example["image_index"]),
                int(card["feature_id"]),
                device,
            )
            boxed, zoom, region = boxed_feature_views(config, image, response)
            boxed_path = feature_root / f"{prefix}_boxed.png"
            zoom_path = feature_root / f"{prefix}_zoom.png"
            evidence_path = feature_root / f"{prefix}_evidence.png"
            boxed.save(boxed_path)
            zoom.save(zoom_path)
            feature_evidence_triptych(
                image,
                boxed,
                zoom,
                group_name,
                float(example["activation"]),
            ).save(evidence_path)
            paths.append(evidence_path)
            manifest.append(
                {
                    "id": prefix,
                    "group": group_name,
                    "activation": float(example["activation"]),
                    "activation_band": example.get("activation_band"),
                    "class_name": example["class_name"],
                    "paths": [str(evidence_path), str(boxed_path), str(zoom_path)],
                    "region": region,
                }
            )
    card["individual_evidence"] = {"discovery": manifest}
    return paths


def render_individual_validation_evidence(
    config: Config,
    bundle: DatasetBundle,
    card: dict[str, Any],
    root: str | Path,
) -> list[Path]:
    dataset = select_split(bundle, str(card["cache_split"]))
    feature_root = Path(root) / f"feature_{int(card['feature_id']):05d}" / "validation"
    feature_root.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = []
    for tile in card["blind_validation_tiles"]:
        example = tile["example"]
        sample = dataset[int(example["dataset_index"])]
        image = _to_display_image(sample["image"], config.data.name).convert("RGB")
        path = feature_root / f"{tile['tile_id']}_original.png"
        image.resize((256, 256), Image.Resampling.NEAREST).save(path)
        paths.append(path)
        manifest.append(
            {
                "tile_id": tile["tile_id"],
                "path": str(path),
                "class_name": example["class_name"],
            }
        )
    card.setdefault("individual_evidence", {})["validation"] = manifest
    return paths


def boxed_feature_views(
    config: Config,
    image: Image.Image,
    response: np.ndarray,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    bbox, peak, active_fraction = strongest_response_region(config, response)
    width, height = image.size
    y0, x0, y1, x1 = bbox
    scale_x = width / response.shape[1]
    scale_y = height / response.shape[0]
    pixel_box = [
        int(np.floor(x0 * scale_x)),
        int(np.floor(y0 * scale_y)),
        int(np.ceil(x1 * scale_x)),
        int(np.ceil(y1 * scale_y)),
    ]
    minimum = int(round(min(width, height) * config.evaluation.region_minimum_fraction))
    pixel_box = expand_minimum_box(pixel_box, width, height, minimum)
    padding = int(round(min(width, height) * config.evaluation.region_padding_fraction))
    pixel_box = [
        max(0, pixel_box[0] - padding),
        max(0, pixel_box[1] - padding),
        min(width, pixel_box[2] + padding),
        min(height, pixel_box[3] + padding),
    ]
    boxed = image.resize((256, 256), Image.Resampling.NEAREST)
    boxed_draw = ImageDraw.Draw(boxed)
    display_box = [
        int(round(pixel_box[0] * 256 / width)),
        int(round(pixel_box[1] * 256 / height)),
        int(round(pixel_box[2] * 256 / width)),
        int(round(pixel_box[3] * 256 / height)),
    ]
    for offset in range(4):
        boxed_draw.rectangle(
            [
                display_box[0] - offset,
                display_box[1] - offset,
                display_box[2] + offset,
                display_box[3] + offset,
            ],
            outline=(255, 40, 160),
            width=1,
        )
    crop = image.crop(tuple(pixel_box))
    zoom = Image.new("RGB", (256, 256), (12, 12, 16))
    resized = crop.resize((232, 232), Image.Resampling.NEAREST)
    zoom.paste(resized, (12, 12))
    zoom_draw = ImageDraw.Draw(zoom)
    zoom_draw.rectangle((8, 8, 247, 247), outline=(255, 40, 160), width=4)
    return boxed, zoom, {
        "response_box": [int(value) for value in bbox],
        "pixel_box": pixel_box,
        "peak": peak,
        "active_fraction": active_fraction,
    }


def feature_evidence_triptych(
    image: Image.Image,
    boxed: Image.Image,
    zoom: Image.Image,
    group_name: str,
    activation: float,
) -> Image.Image:
    panel_size = 192
    header_height = 48
    canvas = Image.new("RGB", (panel_size * 3, panel_size + header_height), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    original = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.NEAREST)
    boxed_panel = boxed.resize((panel_size, panel_size), Image.Resampling.NEAREST)
    zoom_panel = zoom.resize((panel_size, panel_size), Image.Resampling.NEAREST)
    canvas.paste(original, (0, header_height))
    canvas.paste(boxed_panel, (panel_size, header_height))
    canvas.paste(zoom_panel, (panel_size * 2, header_height))
    draw.text((8, 6), f"{group_name}  activation={activation:.4f}", fill=(245, 245, 245))
    draw.text((8, 27), "FULL DRAWING", fill=(180, 180, 190))
    draw.text((panel_size + 8, 27), "ACTIVATION BOX", fill=(255, 40, 160))
    draw.text((panel_size * 2 + 8, 27), "LOCAL CROP", fill=(255, 40, 160))
    return canvas


def strongest_response_region(
    config: Config,
    response: np.ndarray,
) -> tuple[tuple[int, int, int, int], float, float]:
    peak = float(response.max())
    if peak <= 0:
        return (0, 0, response.shape[0], response.shape[1]), 0.0, 0.0
    mask = response >= peak * config.evaluation.region_relative_threshold
    labels, count = ndimage.label(mask)
    if count == 0:
        location = np.unravel_index(int(np.argmax(response)), response.shape)
        return (location[0], location[1], location[0] + 1, location[1] + 1), peak, 0.0
    weights = ndimage.sum(response, labels, range(1, count + 1))
    component = int(np.argmax(weights)) + 1
    ys, xs = np.nonzero(labels == component)
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
    return bbox, peak, float(mask.mean())


def expand_minimum_box(
    box: list[int],
    width: int,
    height: int,
    minimum: int,
) -> list[int]:
    x0, y0, x1, y1 = box
    if x1 - x0 < minimum:
        center = (x0 + x1) / 2
        x0 = max(0, int(np.floor(center - minimum / 2)))
        x1 = min(width, x0 + minimum)
        x0 = max(0, x1 - minimum)
    if y1 - y0 < minimum:
        center = (y0 + y1) / 2
        y0 = max(0, int(np.floor(center - minimum / 2)))
        y1 = min(height, y0 + minimum)
        y0 = max(0, y1 - minimum)
    return [x0, y0, x1, y1]


@torch.no_grad()
def feature_response_map(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    image_index: int,
    feature_id: int,
    device: torch.device,
) -> np.ndarray:
    start = int(cache.image_offsets[image_index])
    end = int(cache.image_offsets[image_index + 1])
    inputs = torch.from_numpy(
        np.array(cache.activations[start:end], dtype=np.float32, copy=True)
    ).to(device)
    response = sae.encode(inputs, inference=True)[:, feature_id].detach().cpu().numpy()
    height = int(cache.metadata["spatial_height"])
    width = int(cache.metadata["spatial_width"])
    response_map = np.zeros((height, width), dtype=np.float32)
    for local_index, row_index in enumerate(range(start, end)):
        y = int(cache.spatial_y[row_index])
        x = int(cache.spatial_x[row_index])
        if y >= 0 and x >= 0:
            response_map[y, x] = max(response_map[y, x], float(response[local_index]))
    return response_map


def response_overlay(image: Image.Image, response: np.ndarray) -> Image.Image:
    maximum = float(response.max())
    if maximum <= 0:
        return image.copy()
    normalized = np.clip(response / maximum, 0, 1)
    heat = Image.fromarray(np.uint8(normalized * 255), mode="L").resize(
        image.size,
        Image.Resampling.BILINEAR,
    )
    red = Image.new("RGB", image.size, (255, 48, 24))
    return Image.composite(red, image, heat.point(lambda value: int(value * 0.65)))


def _to_display_image(tensor: torch.Tensor, dataset_name: str) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if dataset_name == "cub200":
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = tensor * std + mean
    else:
        tensor = tensor * 0.5 + 0.5
    return TF.to_pil_image(tensor.clamp(0, 1))
