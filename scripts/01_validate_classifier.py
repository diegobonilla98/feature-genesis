import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.training.checkpoints import read_json
from feature_genesis.training.quality import build_classifier_gate


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
MINIMUM_ACCURACY = 0.75
MINIMUM_BALANCED_ACCURACY = 0.74
MINIMUM_WORST_DECILE_ACCURACY = 0.45
MINIMUM_TOP5_ACCURACY = 0.92
MAXIMUM_VALIDATION_TEST_GAP = 0.015


def main():
    config = load_config(CONFIG_PATH)
    run_dir = Path(config.project.run_dir)
    result = read_json(run_dir / "result.json")
    gate = build_classifier_gate(
        result,
        MINIMUM_ACCURACY,
        MINIMUM_BALANCED_ACCURACY,
        MINIMUM_WORST_DECILE_ACCURACY,
        MINIMUM_TOP5_ACCURACY,
        MAXIMUM_VALIDATION_TEST_GAP,
    )
    path = run_dir / "stage_01_classifier_gate.json"
    path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    if not gate["passed"]:
        raise RuntimeError("Classifier quality gate failed; do not train an SAE")


if __name__ == "__main__":
    main()
