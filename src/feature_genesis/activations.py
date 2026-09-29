from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from feature_genesis.config import Config
from feature_genesis.data.factory import build_datasets, eval_loader, select_split
from feature_genesis.determinism import resolve_device
from feature_genesis.hashing import state_dict_hash, stable_seed
from feature_genesis.models.hooks import ActivationCapture
from feature_genesis.progress import progress
from feature_genesis.training.replay import reconstruct_state


@dataclass
class ActivationCache:
    root: Path
    activations: np.ndarray
    image_ids: np.ndarray
    dataset_indices: np.ndarray
    labels: np.ndarray
    spatial_y: np.ndarray
    spatial_x: np.ndarray
    image_offsets: np.ndarray
    image_attributes: np.ndarray
    image_attribute_certainty: np.ndarray | None
    image_parts: np.ndarray
    image_part_visible: np.ndarray
    predictions: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path, mmap_mode: str = "r") -> "ActivationCache":
        root = Path(root)
        certainty_path = root / "image_attribute_certainty.npy"
        return cls(
            root=root,
            activations=np.load(root / "activations.npy", mmap_mode=mmap_mode),
            image_ids=np.load(root / "image_ids.npy", mmap_mode=mmap_mode),
            dataset_indices=np.load(root / "dataset_indices.npy", mmap_mode=mmap_mode),
            labels=np.load(root / "labels.npy", mmap_mode=mmap_mode),
            spatial_y=np.load(root / "spatial_y.npy", mmap_mode=mmap_mode),
            spatial_x=np.load(root / "spatial_x.npy", mmap_mode=mmap_mode),
            image_offsets=np.load(root / "image_offsets.npy", mmap_mode=mmap_mode),
            image_attributes=np.load(root / "image_attributes.npy", mmap_mode=mmap_mode),
            image_attribute_certainty=np.load(certainty_path, mmap_mode=mmap_mode)
            if certainty_path.exists()
            else None,
            image_parts=np.load(root / "image_parts.npy", mmap_mode=mmap_mode),
            image_part_visible=np.load(root / "image_part_visible.npy", mmap_mode=mmap_mode),
            predictions=np.load(root / "predictions.npy", mmap_mode=mmap_mode),
            metadata=json.loads((root / "metadata.json").read_text(encoding="utf-8")),
        )

    def batches(self, batch_size: int, seed: int, steps: int | None = None) -> Iterator[torch.Tensor]:
        count = len(self.activations)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(count, generator=generator)
        yielded = 0
        while steps is None or yielded < steps:
            for start in range(0, count, batch_size):
                if steps is not None and yielded >= steps:
                    return
                indices = order[start : start + batch_size].numpy()
                if len(indices) < batch_size:
                    generator = torch.Generator().manual_seed(stable_seed(seed, "activation-reshuffle", yielded))
                    extra = torch.randint(0, count, (batch_size - len(indices),), generator=generator).numpy()
                    indices = np.concatenate([indices, extra])
                yield torch.from_numpy(np.array(self.activations[indices], dtype=np.float32, copy=True))
                yielded += 1
            generator = torch.Generator().manual_seed(stable_seed(seed, "activation-epoch", yielded))
            order = torch.randperm(count, generator=generator)


def cache_checkpoint_activations(
    config: Config,
    checkpoint_step: int,
    split: str | None = None,
    max_images: int | None = None,
    spatial_tokens_per_image: int | None = None,
    replay_config: Config | None = None,
) -> Path:
    resolved_replay_config = config if replay_config is None else replay_config
    resolved_split = config.activations.split if split is None else split
    resolved_max_images = config.activations.max_images if max_images is None else max_images
    resolved_spatial_tokens = (
        config.activations.random_spatial_tokens_per_image
        if spatial_tokens_per_image is None
        else spatial_tokens_per_image
    )
    training_bundle = build_datasets(
        resolved_replay_config,
        include_annotations=False,
        canonical_train=False,
    )
    model, _, _, _ = reconstruct_state(
        resolved_replay_config,
        checkpoint_step,
        bundle=training_bundle,
    )
    bundle = build_datasets(
        resolved_replay_config,
        include_annotations=True,
        canonical_train=True,
    )
    device = resolve_device(resolved_replay_config.project.device)
    dataset = select_split(bundle, resolved_split)
    loader = eval_loader(dataset, config, resolved_max_images, config.activations.batch_size)
    iterator = iter(loader)
    first_batch = next(iterator)
    model.eval()
    with ActivationCapture(model, config.model.hook_layer, detach=True) as capture:
        first_result = _forward_batch(model, capture, first_batch, device)
        activation = first_result[0]
        if activation.ndim == 4:
            height, width = activation.shape[2], activation.shape[3]
            available_tokens = height * width
            tokens_per_image = resolved_spatial_tokens or available_tokens
            tokens_per_image = min(tokens_per_image, available_tokens)
            feature_dim = activation.shape[1]
        elif activation.ndim == 2:
            height, width = 1, 1
            tokens_per_image = 1
            feature_dim = activation.shape[1]
        else:
            raise ValueError(f"Unsupported activation shape: {activation.shape}")
        total_images = len(loader.dataset)
        total_rows = total_images * tokens_per_image
        root = Path(config.project.run_dir) / config.activations.cache_dir / f"step_{checkpoint_step:08d}_{resolved_split}"
        root.mkdir(parents=True, exist_ok=True)
        dtype = np.float16 if config.activations.dtype == "float16" else np.float32
        arrays = _allocate_arrays(root, total_rows, total_images, feature_dim, first_batch, dtype)
        row_cursor = 0
        image_cursor = 0
        batches = itertools.chain(
            [(first_batch, first_result)],
            ((_batch, _forward_batch(model, capture, _batch, device)) for _batch in iterator),
        )
        for batch, result in progress(batches, total=len(loader), desc=f"cache step {checkpoint_step}"):
            activation, logits = result
            rows, ys, xs = _select_rows(
                activation,
                batch["image_id"],
                tokens_per_image,
                config.project.seed,
            )
            batch_images = batch["image"].shape[0]
            batch_rows = rows.shape[0]
            arrays["activations"][row_cursor : row_cursor + batch_rows] = rows.detach().cpu().numpy().astype(dtype, copy=False)
            repeated_image_ids = np.repeat(batch["image_id"].numpy(), tokens_per_image)
            repeated_indices = np.repeat(batch["dataset_index"].numpy(), tokens_per_image)
            repeated_labels = np.repeat(batch["label"].numpy(), tokens_per_image)
            arrays["image_ids"][row_cursor : row_cursor + batch_rows] = repeated_image_ids
            arrays["dataset_indices"][row_cursor : row_cursor + batch_rows] = repeated_indices
            arrays["labels"][row_cursor : row_cursor + batch_rows] = repeated_labels
            arrays["spatial_y"][row_cursor : row_cursor + batch_rows] = ys
            arrays["spatial_x"][row_cursor : row_cursor + batch_rows] = xs
            arrays["image_attributes"][image_cursor : image_cursor + batch_images] = batch["attributes"].numpy()
            arrays["image_attribute_certainty"][
                image_cursor : image_cursor + batch_images
            ] = batch["attribute_certainty"].numpy()
            arrays["image_parts"][image_cursor : image_cursor + batch_images] = batch["parts"].numpy()
            arrays["image_part_visible"][image_cursor : image_cursor + batch_images] = batch["part_visible"].numpy()
            arrays["predictions"][image_cursor : image_cursor + batch_images] = logits.argmax(dim=-1).detach().cpu().numpy()
            arrays["image_offsets"][image_cursor : image_cursor + batch_images] = np.arange(
                row_cursor, row_cursor + batch_rows, tokens_per_image, dtype=np.int64
            )
            row_cursor += batch_rows
            image_cursor += batch_images
        arrays["image_offsets"][total_images] = total_rows
    metadata = {
        "checkpoint_step": checkpoint_step,
        "model_sha256": state_dict_hash(model.state_dict()),
        "layer": config.model.hook_layer,
        "split": resolved_split,
        "images": total_images,
        "rows": total_rows,
        "feature_dim": feature_dim,
        "spatial_height": height,
        "spatial_width": width,
        "tokens_per_image": tokens_per_image,
        "dtype": config.activations.dtype,
        "dataset_signature": bundle.signature,
        "attribute_names": bundle.attribute_names,
        "attribute_certainty_available": bool(bundle.attribute_names),
        "part_names": bundle.part_names,
        "class_names": bundle.class_names,
    }
    for array in arrays.values():
        array.flush()
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _forward_batch(model, capture: ActivationCapture, batch: dict[str, torch.Tensor], device: torch.device):
    with torch.no_grad():
        logits = model(batch["image"].to(device, non_blocking=True))
    if capture.value is None:
        raise RuntimeError("Activation hook did not fire")
    return capture.value, logits


def _select_rows(
    activation: torch.Tensor,
    image_ids: torch.Tensor,
    tokens_per_image: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if activation.ndim == 2:
        size = activation.shape[0]
        return activation, np.full(size, -1, dtype=np.int16), np.full(size, -1, dtype=np.int16)
    batch, channels, height, width = activation.shape
    rows = activation.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    selected_rows = []
    selected_y = []
    selected_x = []
    for batch_index, image_id in enumerate(image_ids.tolist()):
        if tokens_per_image == height * width:
            indices = torch.arange(height * width, device=activation.device)
        else:
            generator = torch.Generator().manual_seed(stable_seed(seed, "spatial-token", image_id))
            indices = torch.randperm(height * width, generator=generator)[:tokens_per_image].to(activation.device)
        selected_rows.append(rows[batch_index, indices])
        selected_y.extend((indices // width).detach().cpu().tolist())
        selected_x.extend((indices % width).detach().cpu().tolist())
    return torch.cat(selected_rows, dim=0), np.asarray(selected_y, dtype=np.int16), np.asarray(selected_x, dtype=np.int16)


def _allocate_arrays(root: Path, rows: int, images: int, feature_dim: int, first_batch: dict[str, torch.Tensor], dtype):
    attribute_dim = first_batch["attributes"].shape[1]
    part_count = first_batch["parts"].shape[1]
    return {
        "activations": np.lib.format.open_memmap(root / "activations.npy", mode="w+", dtype=dtype, shape=(rows, feature_dim)),
        "image_ids": np.lib.format.open_memmap(root / "image_ids.npy", mode="w+", dtype=np.int64, shape=(rows,)),
        "dataset_indices": np.lib.format.open_memmap(root / "dataset_indices.npy", mode="w+", dtype=np.int64, shape=(rows,)),
        "labels": np.lib.format.open_memmap(root / "labels.npy", mode="w+", dtype=np.int64, shape=(rows,)),
        "spatial_y": np.lib.format.open_memmap(root / "spatial_y.npy", mode="w+", dtype=np.int16, shape=(rows,)),
        "spatial_x": np.lib.format.open_memmap(root / "spatial_x.npy", mode="w+", dtype=np.int16, shape=(rows,)),
        "image_offsets": np.lib.format.open_memmap(root / "image_offsets.npy", mode="w+", dtype=np.int64, shape=(images + 1,)),
        "image_attributes": np.lib.format.open_memmap(root / "image_attributes.npy", mode="w+", dtype=np.float32, shape=(images, attribute_dim)),
        "image_attribute_certainty": np.lib.format.open_memmap(
            root / "image_attribute_certainty.npy",
            mode="w+",
            dtype=np.float32,
            shape=(images, attribute_dim),
        ),
        "image_parts": np.lib.format.open_memmap(root / "image_parts.npy", mode="w+", dtype=np.float32, shape=(images, part_count, 2)),
        "image_part_visible": np.lib.format.open_memmap(root / "image_part_visible.npy", mode="w+", dtype=np.bool_, shape=(images, part_count)),
        "predictions": np.lib.format.open_memmap(root / "predictions.npy", mode="w+", dtype=np.int64, shape=(images,)),
    }
