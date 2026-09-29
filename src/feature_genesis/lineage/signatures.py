from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from feature_genesis.activations import ActivationCache
from feature_genesis.hashing import stable_seed
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@dataclass
class FeatureSignatures:
    decoder: np.ndarray
    activation_embedding: np.ndarray
    concept_profile: np.ndarray
    frequency: np.ndarray
    mean_when_active: np.ndarray
    image_top_indices: np.ndarray
    image_top_values: np.ndarray
    probe_row_indices: np.ndarray
    metadata: dict[str, Any]

    def save(self, root: str | Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "signatures.npz",
            decoder=self.decoder,
            activation_embedding=self.activation_embedding,
            concept_profile=self.concept_profile,
            frequency=self.frequency,
            mean_when_active=self.mean_when_active,
            image_top_indices=self.image_top_indices,
            image_top_values=self.image_top_values,
            probe_row_indices=self.probe_row_indices,
        )
        (root / "metadata.json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")
        return root

    @classmethod
    def load(cls, root: str | Path) -> "FeatureSignatures":
        root = Path(root)
        data = np.load(root / "signatures.npz", allow_pickle=False)
        return cls(
            decoder=data["decoder"],
            activation_embedding=data["activation_embedding"],
            concept_profile=data["concept_profile"],
            frequency=data["frequency"],
            mean_when_active=data["mean_when_active"],
            image_top_indices=data["image_top_indices"],
            image_top_values=data["image_top_values"],
            probe_row_indices=data["probe_row_indices"],
            metadata=json.loads((root / "metadata.json").read_text(encoding="utf-8")),
        )


@torch.no_grad()
def compute_feature_signatures(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    device: torch.device,
    batch_size: int,
    seed: int,
    probe_rows: int = 4096,
    projection_dim: int = 256,
    top_images: int = 32,
) -> FeatureSignatures:
    sae.eval()
    dictionary_size = sae.dictionary_size
    count = len(cache.activations)
    generator = torch.Generator().manual_seed(stable_seed(seed, "signature-probes", count))
    probe_indices = torch.randperm(count, generator=generator)[: min(probe_rows, count)].numpy()
    probe_inputs = torch.from_numpy(np.array(cache.activations[probe_indices], dtype=np.float32, copy=True)).to(device)
    probe_codes = _encode_chunks(sae, probe_inputs, batch_size)
    centered = probe_codes - probe_codes.mean(dim=0, keepdim=True)
    centered = centered / centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    projection_generator = torch.Generator(device=device).manual_seed(stable_seed(seed, "signature-projection"))
    projection = torch.randint(
        0,
        2,
        (centered.shape[0], projection_dim),
        generator=projection_generator,
        device=device,
        dtype=torch.int8,
    ).float()
    projection = (projection * 2 - 1) / projection_dim**0.5
    activation_embedding = F.normalize(centered.T @ projection, dim=-1)
    image_codes = _image_max_codes(sae, cache, device, batch_size)
    active_counts = torch.zeros(dictionary_size, dtype=torch.float64, device=device)
    activation_sums = torch.zeros(dictionary_size, dtype=torch.float64, device=device)
    for start in progress(range(0, count, batch_size), desc="signature stats"):
        inputs = torch.from_numpy(np.array(cache.activations[start : start + batch_size], dtype=np.float32, copy=True)).to(device)
        codes = sae.encode(inputs, inference=True)
        active_counts += codes.ne(0).sum(dim=0).double()
        activation_sums += codes.sum(dim=0).double()
    frequency = active_counts / count
    mean_when_active = activation_sums / active_counts.clamp_min(1)
    concept = _concept_profile(image_codes, cache)
    top_count = min(top_images, image_codes.shape[0])
    top_values, top_indices = torch.topk(image_codes.T, top_count, dim=-1)
    metadata = {
        "dictionary_size": dictionary_size,
        "input_dim": sae.input_dim,
        "images": int(image_codes.shape[0]),
        "rows": count,
        "probe_rows": len(probe_indices),
        "projection_dim": projection_dim,
        "attribute_count": int(cache.image_attributes.shape[1]),
        "class_count": len(cache.metadata.get("class_names", [])),
        "checkpoint_step": cache.metadata["checkpoint_step"],
    }
    return FeatureSignatures(
        decoder=sae.decoder_directions().detach().cpu().numpy().astype(np.float32),
        activation_embedding=activation_embedding.cpu().numpy().astype(np.float32),
        concept_profile=concept.cpu().numpy().astype(np.float32),
        frequency=frequency.float().cpu().numpy(),
        mean_when_active=mean_when_active.float().cpu().numpy(),
        image_top_indices=top_indices.cpu().numpy().astype(np.int32),
        image_top_values=top_values.cpu().numpy().astype(np.float32),
        probe_row_indices=probe_indices.astype(np.int64),
        metadata=metadata,
    )


@torch.no_grad()
def _encode_chunks(sae: SparseAutoencoder, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    output = []
    for start in range(0, len(inputs), batch_size):
        output.append(sae.encode(inputs[start : start + batch_size], inference=True))
    return torch.cat(output, dim=0)


@torch.no_grad()
def _image_max_codes(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    image_count = len(cache.image_offsets) - 1
    output = torch.zeros((image_count, sae.dictionary_size), device=device)
    images_per_group = max(1, batch_size // max(1, int(cache.metadata["tokens_per_image"])))
    for image_start in progress(range(0, image_count, images_per_group), desc="signature image codes"):
        image_end = min(image_count, image_start + images_per_group)
        row_start = int(cache.image_offsets[image_start])
        row_end = int(cache.image_offsets[image_end])
        inputs = torch.from_numpy(np.array(cache.activations[row_start:row_end], dtype=np.float32, copy=True)).to(device)
        codes = _encode_chunks(sae, inputs, batch_size)
        for image_index in range(image_start, image_end):
            local_start = int(cache.image_offsets[image_index]) - row_start
            local_end = int(cache.image_offsets[image_index + 1]) - row_start
            output[image_index] = codes[local_start:local_end].amax(dim=0)
    return output


def _concept_profile(image_codes: torch.Tensor, cache: ActivationCache) -> torch.Tensor:
    attributes = torch.from_numpy(np.array(cache.image_attributes, dtype=np.float32, copy=True)).to(image_codes.device)
    attribute_effect = _binary_effects(image_codes, attributes)
    labels = torch.from_numpy(np.array(cache.labels[cache.image_offsets[:-1]], dtype=np.int64, copy=True)).to(image_codes.device)
    class_count = len(cache.metadata.get("class_names", []))
    if class_count == 0:
        class_count = int(labels.max().item()) + 1
    classes = F.one_hot(labels, class_count).float()
    class_effect = _binary_effects(image_codes, classes)
    return F.normalize(torch.cat([attribute_effect, class_effect], dim=1), dim=-1)


def _binary_effects(image_codes: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
    positive_count = indicators.sum(dim=0).clamp_min(1)
    negative_count = (1 - indicators).sum(dim=0).clamp_min(1)
    positive_mean = indicators.T @ image_codes / positive_count[:, None]
    negative_mean = (1 - indicators).T @ image_codes / negative_count[:, None]
    scale = image_codes.std(dim=0, keepdim=True).clamp_min(1e-6)
    return ((positive_mean - negative_mean) / scale).T
