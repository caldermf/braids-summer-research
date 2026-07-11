from __future__ import annotations

import math
from collections.abc import Sequence


def cooled_temperature(
    initial_temperature: float,
    minimum_temperature: float,
    cooling_rate: float,
    round_index: int,
    boost: float = 1.0,
    maximum_boost: float = 4.0,
) -> float:
    """Return the depth-cooled temperature with an optional reheat boost."""
    if initial_temperature <= 0 or minimum_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if not 0 < cooling_rate <= 1:
        raise ValueError("cooling_rate must lie in (0, 1]")
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")
    if boost < 1 or maximum_boost < 1:
        raise ValueError("temperature boosts must be at least one")

    base = max(
        minimum_temperature,
        initial_temperature * cooling_rate**round_index,
    )
    return min(initial_temperature * maximum_boost, base * min(boost, maximum_boost))


def boltzmann_bucket_weights(
    energies: Sequence[int | float], temperature: float
) -> list[float]:
    """Assign exp(-(energy - minimum_energy) / temperature) to each bucket."""
    if not energies:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    minimum = min(float(value) for value in energies)
    return [
        math.exp(max(-745.0, -(float(value) - minimum) / temperature))
        for value in energies
    ]


def _capped_weighted_allocation(
    capacities: Sequence[int], weights: Sequence[float], budget: int
) -> list[int]:
    """Allocate an integer budget proportionally, redistributing capped shares."""
    allocations = [0] * len(capacities)
    remaining = min(budget, sum(capacities))

    while remaining > 0:
        active = [
            index
            for index, capacity in enumerate(capacities)
            if allocations[index] < capacity
        ]
        if not active:
            break
        total_weight = sum(weights[index] for index in active)
        if total_weight <= 0:
            active_weights = {index: 1.0 for index in active}
            total_weight = float(len(active))
        else:
            active_weights = {index: weights[index] for index in active}

        raw_shares = {
            index: remaining * active_weights[index] / total_weight
            for index in active
        }
        additions = {
            index: min(
                capacities[index] - allocations[index],
                int(math.floor(raw_shares[index])),
            )
            for index in active
        }
        added = sum(additions.values())

        if added == 0:
            ordering = sorted(
                active,
                key=lambda index: (
                    raw_shares[index] - math.floor(raw_shares[index]),
                    active_weights[index],
                    -index,
                ),
                reverse=True,
            )
            for index in ordering[:remaining]:
                additions[index] = 1
            added = sum(additions.values())

        for index, addition in additions.items():
            allocations[index] += addition
        remaining -= added

    return allocations


def allocate_annealed_quotas(
    energies: Sequence[int | float],
    counts: Sequence[int],
    budget: int,
    temperature: float,
    minimum_per_bucket: int = 0,
) -> list[int]:
    """
    Allocate parent slots across projlen buckets using Boltzmann weights.

    Every stored braid in an author bucket is already a uniform reservoir
    sample. Subsampling it uniformly to the returned quota therefore preserves
    that property while making the effective bucket sizes temperature-dependent.
    """
    if len(energies) != len(counts):
        raise ValueError("energies and counts must have the same length")
    if budget < 0 or minimum_per_bucket < 0:
        raise ValueError("budget and minimum_per_bucket must be nonnegative")
    if any(count < 0 for count in counts):
        raise ValueError("bucket counts must be nonnegative")
    if not counts or budget == 0:
        return [0] * len(counts)

    target = min(budget, sum(counts))
    weights = boltzmann_bucket_weights(energies, temperature)
    requested_floors = [min(count, minimum_per_bucket) for count in counts]

    if sum(requested_floors) <= target:
        quotas = requested_floors
    else:
        quotas = _capped_weighted_allocation(requested_floors, weights, target)

    remaining = target - sum(quotas)
    if remaining:
        residual_capacities = [count - quota for count, quota in zip(counts, quotas)]
        extras = _capped_weighted_allocation(
            residual_capacities,
            weights,
            remaining,
        )
        quotas = [quota + extra for quota, extra in zip(quotas, extras)]

    return quotas


def allocate_core_annealed_quotas(
    energies: Sequence[int | float],
    counts: Sequence[int],
    budget: int,
    temperature: float,
    core_fraction: float = 0.95,
    minimum_per_bucket: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """
    Preserve a hard low-energy core and anneal the remaining parent budget.

    The core protects the behavior that made the paper reservoir successful.
    The spillover is the experimental component: it dynamically gives some
    capacity to higher-projlen buckets without replacing the core search.
    Returns total, core, and spillover quotas in the input order.
    """
    if len(energies) != len(counts):
        raise ValueError("energies and counts must have the same length")
    if not 0 <= core_fraction <= 1:
        raise ValueError("core_fraction must lie in [0, 1]")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if any(count < 0 for count in counts):
        raise ValueError("bucket counts must be nonnegative")

    target = min(budget, sum(counts))
    core_budget = min(target, int(math.floor(target * core_fraction)))
    core = [0] * len(counts)
    remaining_core = core_budget
    for index in sorted(range(len(energies)), key=lambda item: (energies[item], item)):
        take = min(counts[index], remaining_core)
        core[index] = take
        remaining_core -= take
        if remaining_core == 0:
            break

    residual_counts = [count - quota for count, quota in zip(counts, core)]
    spillover = allocate_annealed_quotas(
        energies=energies,
        counts=residual_counts,
        budget=target - sum(core),
        temperature=temperature,
        minimum_per_bucket=minimum_per_bucket,
    )
    total = [core_quota + extra for core_quota, extra in zip(core, spillover)]
    return total, core, spillover
