import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.training.quality import require_classifier_gate
from feature_genesis.training.checkpoints import list_anchors
from feature_genesis.training.replay import verify_anchor_replay

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
TARGET_STEP = None


def main():
    config = load_config(CONFIG_PATH)
    require_classifier_gate(config.project.run_dir)
    anchors = list_anchors(config.project.run_dir)
    target = TARGET_STEP or anchors[-1][0]
    result = verify_anchor_replay(config, target)
    path = Path(config.project.run_dir) / "replay_verification.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
