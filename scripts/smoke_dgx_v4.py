import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.training.replay import verify_anchor_replay
from feature_genesis.training.trainer import train_project


CONFIG_PATH = Path("configs/quickdraw_dgx_v4_smoke.yaml")
TARGET_STEP = 8


def main():
    config = load_config(CONFIG_PATH)
    training = train_project(config, overwrite=True, resume=False)
    replay = verify_anchor_replay(config, TARGET_STEP)
    result = {"training": training, "replay": replay}
    path = Path(config.project.run_dir) / "smoke_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    if not replay["bitwise_equal"]:
        raise RuntimeError("DGX BF16 replay is not bitwise exact")


if __name__ == "__main__":
    main()
