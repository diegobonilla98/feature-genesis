from __future__ import annotations

import os
from pathlib import Path

LEGACY_CUB_ROOT = Path("data") / "CUB_200_2011"


def data_home() -> Path:
    return Path(os.environ.get("HF_HOME", "data"))


def cub_dataset_dir() -> Path:
    return data_home() / "CUB_200_2011"


def resolve_cub_root(root: str | Path) -> str:
    normalized = Path(root).as_posix()
    if normalized == LEGACY_CUB_ROOT.as_posix():
        return str(cub_dataset_dir())
    return str(root)
