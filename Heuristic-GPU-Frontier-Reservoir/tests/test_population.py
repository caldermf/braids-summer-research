import random
from types import SimpleNamespace

from heuristic_gpu_frontier_reservoir.search import Population


def state(score, projlen):
    return SimpleNamespace(score=score, metrics={"projlen": projlen})


def test_confusion_selects_larger_bins_first():
    population = Population("confusion", 8, 10, .25, 20., random.Random(1))
    for item in (state(.1, 9), state(2.1, 20), state(1.1, 4)): population.add(item)
    selected, _ = population.select(2)
    assert [item.score for item in selected] == [2.1, 1.1]


def test_projlen_selects_smaller_values_first():
    population = Population("projlen", 8, 10, .25, 20., random.Random(1))
    for item in (state(.1, 9), state(2.1, 20), state(1.1, 4)): population.add(item)
    selected, _ = population.select(2)
    assert [item.metrics["projlen"] for item in selected] == [4, 9]
