from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.training.trainer import train_project

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
OVERWRITE = False
RESUME = True


def main():
    config = load_config(CONFIG_PATH)
    print(train_project(config, overwrite=OVERWRITE, resume=RESUME))


if __name__ == "__main__":
    main()
