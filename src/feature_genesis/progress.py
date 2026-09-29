from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = True,
) -> Iterator[T]:
    return tqdm(iterable, total=total, desc=desc, leave=leave)
