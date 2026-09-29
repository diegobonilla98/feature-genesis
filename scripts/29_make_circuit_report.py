from pathlib import Path

from feature_genesis.circuits.report import render_circuit_report
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")


def main():
    config = load_config(CONFIG_PATH)
    print(render_circuit_report(config))


if __name__ == "__main__":
    main()
