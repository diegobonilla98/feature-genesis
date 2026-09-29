import json
from pathlib import Path

import torch

from feature_genesis.config import load_config, save_resolved_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.models.factory import build_model
from feature_genesis.models.hooks import ActivationCapture


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAMPLE_BATCH_SIZE = 8


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(config, include_annotations=True, canonical_train=True)
    samples = [bundle.train[index] for index in range(SAMPLE_BATCH_SIZE)]
    images = torch.stack([sample["image"] for sample in samples])
    labels = torch.stack([sample["label"] for sample in samples])
    model = build_model(config, bundle.num_classes)
    with ActivationCapture(model, config.model.hook_layer, detach=True) as capture:
        logits = model(images)
    if capture.value is None:
        raise RuntimeError("QuickDraw activation hook did not fire")
    if logits.shape != (SAMPLE_BATCH_SIZE, bundle.num_classes):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if capture.value.ndim != 4:
        raise RuntimeError(f"Expected a spatial hook tensor, received {tuple(capture.value.shape)}")
    if images.shape[1] != 1:
        raise RuntimeError(f"Expected grayscale input, received {images.shape[1]} channels")
    if int(labels.min()) < 0 or int(labels.max()) >= bundle.num_classes:
        raise RuntimeError("QuickDraw labels fall outside the configured class range")
    run_root = Path(config.project.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_root / "resolved_config.yaml")
    report = {
        "status": "ready",
        "dataset": config.data.name,
        "dataset_root": config.data.root,
        "train_images": len(bundle.train),
        "validation_images": len(bundle.validation),
        "test_images": len(bundle.test),
        "classes": bundle.num_classes,
        "image_shape": list(images.shape[1:]),
        "hook_layer": config.model.hook_layer,
        "hook_shape": list(capture.value.shape[1:]),
        "dictionary_size": config.sae.dictionary_size,
        "sparsity_k": config.sae.k,
        "analysis_steps": config.train.analysis_steps,
    }
    path = run_root / "stage_00_quickdraw_gate.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
