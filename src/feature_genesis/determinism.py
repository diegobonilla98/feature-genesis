from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL
import torch
import torchvision

from feature_genesis.config import ReproSection


@dataclass
class RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


def configure_determinism(seed: int, config: ReproSection) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.strict
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
    torch.use_deterministic_algorithms(config.deterministic_algorithms, warn_only=config.warn_only)


def capture_rng_state() -> RNGState:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return RNGState(random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda)


def _coerce_torch_rng_state(state: Any) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        tensor = state.detach().cpu()
    else:
        tensor = torch.tensor(state, dtype=torch.uint8)
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)
    return tensor.contiguous()


def restore_rng_state(state: RNGState | dict[str, Any]) -> None:
    if isinstance(state, dict):
        state = RNGState(**state)
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(_coerce_torch_rng_state(state.torch_cpu))
    if state.torch_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_coerce_torch_rng_state(item) for item in state.torch_cuda])


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def package_freeze() -> list[str]:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def environment_fingerprint() -> dict[str, Any]:
    output = critical_environment_fingerprint()
    output.update(
        {
            "packages": package_freeze(),
            "git_commit": git_commit(),
        }
    )
    return output


def critical_environment_fingerprint() -> dict[str, Any]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _torchvision_version() -> str:
    return torchvision.__version__
