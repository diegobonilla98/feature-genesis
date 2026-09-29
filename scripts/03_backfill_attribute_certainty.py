import os
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, select_split


CONFIG_PATH = Path("configs/cub_resnet18.yaml")


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(
        config,
        include_annotations=True,
        canonical_train=True,
    )
    dataset = select_split(bundle, config.activations.split)
    records_by_image_id = {
        int(record.image_id): record for record in dataset.records
    }
    outputs = []
    for step in config.activations.checkpoint_steps:
        root = (
            Path(config.project.run_dir)
            / config.activations.cache_dir
            / f"step_{step:08d}_{config.activations.split}"
        )
        cache = ActivationCache.load(root)
        image_ids = np.asarray(
            cache.image_ids[cache.image_offsets[:-1]],
            dtype=np.int64,
        )
        records = [records_by_image_id[int(image_id)] for image_id in image_ids]
        expected_attributes = np.stack(
            [record.attributes for record in records],
        ).astype(np.float32)
        cached_attributes = np.asarray(cache.image_attributes, dtype=np.float32)
        if not np.array_equal(expected_attributes, cached_attributes):
            raise ValueError(f"Cached attributes do not match CUB records at step {step}")
        certainty = np.stack(
            [record.attribute_certainty for record in records],
        ).astype(np.float32)
        temporary_path = root / "image_attribute_certainty.tmp.npy"
        final_path = root / "image_attribute_certainty.npy"
        np.save(temporary_path, certainty)
        os.replace(temporary_path, final_path)
        outputs.append(
            {
                "step": step,
                "path": str(final_path),
                "shape": list(certainty.shape),
                "minimum": float(certainty.min()),
                "maximum": float(certainty.max()),
            }
        )
    print(outputs)


if __name__ == "__main__":
    main()
