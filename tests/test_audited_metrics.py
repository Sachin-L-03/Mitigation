"""Hand-calculated unit tests for audited metric implementations in src/mitigation/metrics.py."""

import pytest
import numpy as np

from src.mitigation.metrics import (
    normalize_text,
    compute_token_f1,
    compute_char_ngram_similarity,
    check_concept_accuracy,
    evaluate_cluster_consistency,
    compute_bootstrap_ci,
    compare_conditions_paired,
    apply_holm_bonferroni,
)


def test_hand_calculated_token_f1():
    """Verify hand-calculated token F1:
    pred = 'the dog barked loudly' -> tokens ['dog', 'barked', 'loudly'] (len 3)
    target = 'dog barked' -> tokens ['dog', 'barked'] (len 2)
    common = 2
    P = 2/3, R = 1.0 -> F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
    """
    f1 = compute_token_f1("the dog barked loudly", "dog barked")
    assert pytest.approx(f1, abs=1e-5) == 0.8

    # Edge cases
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1("hello", "") == 0.0
    assert compute_token_f1("", "world") == 0.0
    assert compute_token_f1("no overlap", "completely different") == 0.0


def test_hand_calculated_concept_accuracy():
    """Verify factual concept matching logic on positive and negative cases."""
    gold = "Glutamate"
    assert check_concept_accuracy("In mammals, glutamate is the main excitatory transmitter.", gold) is True
    assert check_concept_accuracy("In mammals, GABA is the main inhibitory transmitter.", gold) is False

    gold_complex = "Angiotensin-converting enzyme (ACE)"
    assert check_concept_accuracy("ACE catalyzes the conversion of Ang I to Ang II.", gold_complex) is True
    assert check_concept_accuracy("Renin cleaves angiotensinogen.", gold_complex) is False


def test_hand_calculated_exact_match_definitions():
    """Verify that all-identical, canonical, and pairwise exact matches are distinctly calculated.

    Cluster answers:
      original: 'apple'
      v1:       'apple'
      v2:       'banana'

    - all_identical: 0.0 (since 'banana' != 'apple')
    - canonical: mean of [apple vs apple (1.0), banana vs apple (0.0)] = 0.5
    - pairwise: 3 pairs: (orig, v1) -> 1.0, (orig, v2) -> 0.0, (v1, v2) -> 0.0 -> mean = 1/3 = 0.33333
    """
    cluster = {
        "original": "apple",
        "lexical": "apple",
        "syntactic": "banana",
    }
    res = evaluate_cluster_consistency(cluster, canonical_key="original")
    assert res["all_identical_exact_match"] == 0.0
    assert pytest.approx(res["canonical_exact_match"], abs=1e-5) == 0.5
    assert pytest.approx(res["pairwise_exact_match"], abs=1e-5) == 1.0 / 3.0


def test_hand_calculated_holm_bonferroni():
    """Verify Holm-Bonferroni correction:
    Raw p-values: [0.01, 0.04, 0.03]
    Sorted:
      rank 0: 0.01 * 3 = 0.03
      rank 1: 0.03 * 2 = 0.06
      rank 2: 0.04 * 1 = 0.04 -> monotonic step-up to 0.06
    Adjusted: [0.03, 0.06, 0.06]
    """
    raw = [0.01, 0.04, 0.03]
    adj = apply_holm_bonferroni(raw)
    assert adj == [0.03, 0.06, 0.06]


def test_bootstrap_ci_bounds():
    """Verify bootstrap confidence interval contains the sample mean."""
    data = [10.0, 12.0, 11.0, 14.0, 13.0, 15.0]
    mean, lower, upper = compute_bootstrap_ci(data, num_samples=500, ci_level=0.95, seed=42)
    assert lower <= mean <= upper
    assert pytest.approx(mean, abs=1e-4) == float(np.mean(data))
