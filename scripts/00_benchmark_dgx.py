import json
import time
from pathlib import Path

import torch

from feature_genesis.config import load_config
from feature_genesis.data.batch_plan import create_batch_plan
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import configure_determinism, resolve_device
from feature_genesis.models.factory import build_model
from feature_genesis.training.trainer import build_optimizer, train_range


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
BENCHMARK_STEPS = 50
OUTPUT_PATH = Path("/mnt/hdd/QuickDraw/feature_genesis_calibration/v4_throughput.json")


def main():
    config = load_config(CONFIG_PATH)
    configure_determinism(config.project.seed, config.reproducibility)
    bundle = build_datasets(config, include_annotations=False)
    plan = create_batch_plan(
        len(bundle.train),
        config.data.batch_size,
        config.train.epochs,
        config.project.seed,
        BENCHMARK_STEPS,
    )
    device = resolve_device(config.project.device)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rows = train_range(model, optimizer, bundle, plan, config, 0, len(plan))
    elapsed = time.perf_counter() - started
    images = int(plan.sizes.sum())
    result = {
        "steps": len(plan),
        "images": images,
        "seconds": elapsed,
        "images_per_second": images / elapsed,
        "seconds_per_step": elapsed / len(plan),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "initial_loss": rows[0]["loss"],
        "final_loss": rows[-1]["loss"],
        "final_accuracy": rows[-1]["accuracy"],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(OUTPUT_PATH)
    print(result)


if __name__ == "__main__":
    main()
