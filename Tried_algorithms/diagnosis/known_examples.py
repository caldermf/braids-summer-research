from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from peyl.braid_data import simple_factor_id_maps

from .models import KernelCase


KNOWN_P5_LENGTH54_FACTOR_IDS = (
    7, 7, 10, 13, 4, 13, 4, 2, 13, 20, 13, 20, 13, 10, 2, 13, 4, 13,
    4, 13, 7, 21, 20, 13, 20, 13, 10, 16, 16, 2, 13, 4, 13, 4, 13, 21,
    20, 13, 20, 13, 21, 10, 13, 4, 13, 4, 2, 16, 13, 11, 13, 11, 13, 21,
)


EMBEDDED_CASES = {
    "p5_length54": KernelCase(
        name="p5_length54",
        factor_ids=KNOWN_P5_LENGTH54_FACTOR_IDS,
    ),
}


def _factor_ids(payload: dict, n: int) -> tuple[int, ...]:
    if "factor_ids" in payload:
        return tuple(int(value) for value in payload["factor_ids"])
    if "gnf_factors" in payload:
        perm_to_id, _ = simple_factor_id_maps(n)
        return tuple(perm_to_id[tuple(perm)] for perm in payload["gnf_factors"])
    raise ValueError("kernel JSON item must contain factor_ids or gnf_factors")


def load_kernel_cases(
    names: Iterable[str],
    json_paths: Iterable[Path],
    n: int,
    all_json_kernels: bool = False,
) -> list[KernelCase]:
    cases: list[KernelCase] = []
    for name in names:
        try:
            cases.append(EMBEDDED_CASES[name])
        except KeyError as exc:
            choices = ", ".join(sorted(EMBEDDED_CASES))
            raise ValueError(f"unknown embedded kernel {name!r}; choose from {choices}") from exc

    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = payload
        elif "kernel_hits" in payload:
            items = payload["kernel_hits"]
        else:
            items = [payload]
        if not all_json_kernels:
            items = items[:1]
        for index, item in enumerate(items):
            label = item.get("name") or item.get("kernel_type") or "kernel"
            cases.append(
                KernelCase(
                    name=f"{path.stem}_{index}_{label}",
                    factor_ids=_factor_ids(item, n=n),
                    source=str(path),
                )
            )

    if not cases:
        cases.append(EMBEDDED_CASES["p5_length54"])

    unique: dict[tuple[int, ...], KernelCase] = {}
    for case in cases:
        unique.setdefault(case.factor_ids, case)
    return list(unique.values())
