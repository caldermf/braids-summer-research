from pathlib import Path
import random

from last_factor_confusion.factors import FactorTable
from last_factor_confusion.representation import JonesAdapter, JonesSpec, install_peyl


AUTHOR_REPO = Path(__file__).resolve().parents[2]


def test_factor_table_has_22_stable_classes():
    install_peyl(AUTHOR_REPO)
    table = FactorTable.from_peyl(4)
    assert len(table.permutations) == 22
    assert len(table.checksum()) == 64


def test_adapter_matches_native_prefix_evaluator():
    install_peyl(AUTHOR_REPO)
    from peyl.braid import GNF
    from peyl.braidsearch import evaluate_braid_factors
    braid = GNF.sample(4, 4, rand=random.Random(11))
    adapter = JonesAdapter(AUTHOR_REPO, JonesSpec(p=5))
    final_prefix = adapter.evaluate_prefixes([braid])[-1][0]
    native = evaluate_braid_factors(adapter.rep, braid)
    assert (adapter.normalize_image(final_prefix) == adapter.normalize_image(native)).all()
    assert adapter.projlen(native) == adapter.normalize_image(native).shape[-1] - 1

