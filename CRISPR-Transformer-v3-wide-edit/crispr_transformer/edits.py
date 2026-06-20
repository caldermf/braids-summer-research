from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .gnf import GNFAutomaton


@dataclass(frozen=True, order=True)
class EditGeometry:
    start: int
    delete_length: int
    insert_length: int

    @property
    def length_delta(self) -> int:
        return self.insert_length - self.delete_length

    def to_dict(self) -> dict:
        return asdict(self)


def valid_geometries(
    parent_length: int,
    *,
    min_length: int,
    max_length: int,
    max_delete: int,
    max_insert: int,
    max_net_delta: int,
) -> list[EditGeometry]:
    if parent_length < 2:
        return []
    output = []
    for delete_length in range(1, min(max_delete, parent_length - 1) + 1):
        for insert_length in range(1, max_insert + 1):
            delta = insert_length - delete_length
            child_length = parent_length + delta
            if abs(delta) > max_net_delta:
                continue
            if not min_length <= child_length <= max_length:
                continue
            for start in range(parent_length - delete_length + 1):
                output.append(EditGeometry(start, delete_length, insert_length))
    return output


def sample_geometries(
    parent_length: int,
    count: int,
    rng: random.Random,
    **bounds,
) -> list[EditGeometry]:
    actions = valid_geometries(parent_length, **bounds)
    if count <= 0 or count >= len(actions):
        rng.shuffle(actions)
        return actions
    return rng.sample(actions, count)


def sample_balanced_geometries(
    parent_length: int,
    count: int,
    rng: random.Random,
    *,
    min_length: int,
    max_length: int,
    max_delete: int,
    max_insert: int,
    max_net_delta: int,
) -> list[EditGeometry]:
    """Sample broad edit scales and locations without enumerating the action space."""
    if count <= 0 or parent_length < 2:
        return []
    largest_delete = min(max_delete, parent_length - 1)
    scale_bands = [
        (1, min(8, largest_delete)),
        (9, min(24, largest_delete)),
        (25, largest_delete),
    ]
    scale_bands = [(low, high) for low, high in scale_bands if low <= high]
    locations = ("prefix", "interior", "suffix")
    output: list[EditGeometry] = []
    seen = set()
    attempts = 0
    limit = max(1_000, count * 100)
    while len(output) < count and attempts < limit:
        index = attempts
        attempts += 1
        low, high = scale_bands[index % len(scale_bands)]
        delete_length = rng.randint(low, high)
        feasible_deltas = [
            delta
            for delta in range(-max_net_delta, max_net_delta + 1)
            if 1 <= delete_length + delta <= max_insert
            and min_length
            <= parent_length + delta
            <= max_length
        ]
        if not feasible_deltas:
            continue
        delta = rng.choice(feasible_deltas)
        insert_length = delete_length + delta
        max_start = parent_length - delete_length
        location = locations[(index // len(scale_bands)) % len(locations)]
        edge_width = min(max_start, max(2, parent_length // 10))
        if location == "prefix":
            start = rng.randint(0, edge_width)
        elif location == "suffix":
            start = rng.randint(max(0, max_start - edge_width), max_start)
        else:
            start = rng.randint(0, max_start)
        geometry = EditGeometry(start, delete_length, insert_length)
        if geometry in seen:
            continue
        seen.add(geometry)
        output.append(geometry)
    if len(output) < count:
        fallback = valid_geometries(
            parent_length,
            min_length=min_length,
            max_length=max_length,
            max_delete=max_delete,
            max_insert=max_insert,
            max_net_delta=max_net_delta,
        )
        rng.shuffle(fallback)
        for geometry in fallback:
            if geometry not in seen:
                seen.add(geometry)
                output.append(geometry)
            if len(output) >= count:
                break
    if not output:
        raise RuntimeError("no legal wide-edit geometries are available")
    rng.shuffle(output)
    return output


def apply_geometry(
    parent: Sequence[int],
    geometry: EditGeometry,
    automaton: GNFAutomaton,
    rng: random.Random,
    attempts: int = 20,
) -> tuple[int, ...]:
    original = tuple(int(value) for value in parent)
    start = geometry.start
    end = start + geometry.delete_length
    if start < 0 or end > len(original):
        raise ValueError("edit geometry is outside the parent braid")
    left = original[start - 1] if start else None
    right = original[end] if end < len(original) else None
    for _ in range(attempts):
        try:
            replacement = automaton.sample_bridge(
                left=left,
                right=right,
                length=geometry.insert_length,
                rng=rng,
            )
        except ValueError as exc:
            raise RuntimeError(
                "edit geometry has no legal GNF bridge for its boundaries"
            ) from exc
        child = original[:start] + replacement + original[end:]
        if child != original and automaton.is_legal(child):
            return child
    raise RuntimeError("failed to sample a nontrivial legal replacement")


def enumerate_and_apply(
    parent: Sequence[int],
    geometries: Iterable[EditGeometry],
    replacements_per_geometry: int,
    automaton: GNFAutomaton,
    rng: random.Random,
) -> list[tuple[EditGeometry, tuple[int, ...]]]:
    output = []
    seen = {tuple(parent)}
    for geometry in geometries:
        produced = 0
        for _ in range(max(4, replacements_per_geometry * 4)):
            if produced >= replacements_per_geometry:
                break
            try:
                child = apply_geometry(parent, geometry, automaton, rng)
            except (ValueError, RuntimeError):
                continue
            if child in seen:
                continue
            seen.add(child)
            output.append((geometry, child))
            produced += 1
    return output
