"""Check the pool-size expectation against exhaustive small-corpus subsampling."""

from itertools import combinations

import pytest
from pool_size_counterfactual import probability_hit


@pytest.mark.parametrize("retained,k", [(3, 2), (5, 3), (8, 3), (8, 5)])
def test_uniform_shared_pool_agrees_with_exact_enumeration(retained, k):
    # Source 0 is relevant; source 1 belongs to another evaluation query. Both
    # must be retained. Two ordinary negatives and source 1 outrank source 0.
    ordering = [1, 2, 3, 0, 4, 5, 6, 7]
    outcomes = []
    for sampled in combinations(range(2, 8), retained - 2):
        kept = {0, 1, *sampled}
        filtered = [doc for doc in ordering if doc in kept]
        outcomes.append(float(filtered.index(0) < k))
    expected = sum(outcomes) / len(outcomes)
    assert probability_hit(8, 2, retained, random_above=2, forced_above=1, k=k) == pytest.approx(
        expected
    )


def test_forced_competitors_can_prevent_a_hit_even_after_thinning():
    assert probability_hit(100, 12, 12, random_above=0, forced_above=11) == 0
    assert probability_hit(100, 12, 12, random_above=50, forced_above=0) == 1


def test_invalid_pool_size_fails():
    with pytest.raises(ValueError, match="Invalid pool"):
        probability_hit(20, 5, 4, random_above=3, forced_above=0)


def test_pool_with_no_optional_units_is_deterministic():
    assert probability_hit(3, 3, 3, random_above=0, forced_above=1, k=2) == 1
    assert probability_hit(3, 3, 3, random_above=0, forced_above=2, k=2) == 0
