from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.sae.validation import validate_stage_four


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"


def main():
    config = load_config(CONFIG_PATH)
    result = validate_stage_four(config, SAE_KIND)
    print(result["path"])
    if not result["passed"]:
        raise RuntimeError("Stage 4 scientific validation gate failed")


if __name__ == "__main__":
    main()
