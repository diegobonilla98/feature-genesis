import json
from pathlib import Path

from PIL import Image

from feature_genesis.config import load_config
from feature_genesis.data.batch_plan import SampleKey
from feature_genesis.data.factory import build_datasets
from feature_genesis.labeling.cards import _to_display_image
from feature_genesis.sae.io import sae_analysis_root


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
RENDER_BATCHES_PER_DIRECTION = 4


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(config, include_annotations=False)
    attribution_root = sae_analysis_root(config, "attribution", kind=SAE_KIND)
    causal_path = attribution_root / "causal_validation_summary.json"
    causal = (
        json.loads(causal_path.read_text(encoding="utf-8"))
        if causal_path.exists()
        else {"records": []}
    )
    for feature_id in accepted_feature_ids(config):
        render_feature(config, bundle, attribution_root, causal, feature_id)


def render_feature(config, bundle, attribution_root, causal, feature_id):
    feature_root = attribution_root / f"feature_{feature_id:05d}"
    causal_records = {
        (row["predicted_direction"], int(row["candidate_step"])): row
        for row in causal["records"]
        if int(row["feature_id"]) == feature_id
    }
    payload = json.loads(
        (feature_root / "gradient_ranked_batches.json").read_text(encoding="utf-8")
    )
    output_root = feature_root / "training_influence_images"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for direction in ["builders", "breakers"]:
        batches = payload[direction][:RENDER_BATCHES_PER_DIRECTION]
        example_key = "example_builders" if direction == "builders" else "example_breakers"
        for batch_rank, batch in enumerate(batches, start=1):
            causal_record = causal_records.get((direction, int(batch["step"])))
            for example_rank, example in enumerate(batch[example_key], start=1):
                dataset_index = int(example["dataset_index"])
                epoch = int(example["epoch"])
                sample = bundle.train[SampleKey(dataset_index, epoch)]
                image = _to_display_image(sample["image"], config.data.name).convert("RGB")
                path = output_root / (
                    f"{direction}_batch_{batch_rank:02d}_example_{example_rank:02d}"
                    f"_step_{int(batch['step']):08d}_index_{dataset_index:08d}.png"
                )
                image.resize((256, 256), Image.Resampling.NEAREST).save(path)
                manifest.append(
                    {
                        "predicted_direction": direction,
                        "causal_status": (
                            "validated" if causal_record is not None else "candidate_only"
                        ),
                        "causal_role": (
                            causal_record["causal_role"]
                            if causal_record is not None
                            else None
                        ),
                        "predicted_direction_confirmed": (
                            causal_record["predicted_direction_confirmed"]
                            if causal_record is not None
                            else None
                        ),
                        "batch_rank": batch_rank,
                        "example_rank": example_rank,
                        "step": int(batch["step"]),
                        "dataset_index": dataset_index,
                        "epoch": epoch,
                        "label": int(example["label"]),
                        "class_name": bundle.class_names[int(example["label"])],
                        "batch_influence": float(batch["influence"]),
                        "example_influence": float(example["influence"]),
                        "image": str(path),
                    }
                )
    manifest_path = feature_root / "training_influence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(manifest_path)


def accepted_feature_ids(config):
    step = max(config.activations.checkpoint_steps)
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND) / "gemini"
    feature_ids = []
    for path in sorted(root.glob("feature_*_evaluation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["validated_status"] == "accepted":
            feature_ids.append(int(row["feature_id"]))
    return feature_ids


if __name__ == "__main__":
    main()
