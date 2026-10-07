"""Regression tests for evaluator input, tie and FAR-resolution boundaries."""
import numpy as np
import pytest

from aigcfr.eval.verify import Scores, accuracy_at, best_threshold, summarize, tar_at_far


def _inputs(same, diff):
    return np.r_[np.ones(len(same), dtype=int), np.zeros(len(diff), dtype=int)], np.r_[same, diff]


def test_argument_order_and_keyword_calls():
    labels, scores = _inputs([0.9, 0.8], np.linspace(-0.2, 0.3, 1000))
    assert tar_at_far(labels, scores, 0.001) == tar_at_far(labels=labels, scores=scores, far=0.001)
    with pytest.raises(ValueError, match="labels"):
        tar_at_far(scores, labels, 0.001)
    with pytest.raises(ValueError, match="labels"):
        best_threshold(scores, labels)


@pytest.mark.parametrize("labels,scores", [
    ([0, 2], [0.1, 0.8]), ([-1, 1], [0.1, 0.8]),
    ([0, np.nan], [0.1, 0.8]), ([0, np.inf], [0.1, 0.8]),
    (["0", "1"], [0.1, 0.8]), ([0, 1], [0.1, np.nan]),
    ([0, 1], [0.1, np.inf]), ([0, 1], [0.1, -np.inf]),
    ([0, 1], [0.1, 0.8j]), ([0, 1], ["0.1", "0.8"]),
    ([[0, 1]], [[0.1, 0.8]]), ([0, 1], [[0.1, 0.8]]),
    ([0, 1], [0.1]), ([], [0.1]),
])
@pytest.mark.parametrize("function", [best_threshold, tar_at_far, accuracy_at])
def test_malformed_inputs_raise(labels, scores, function):
    with pytest.raises(ValueError):
        if function is best_threshold:
            function(labels, scores)
        else:
            function(labels, scores, 0.5)


@pytest.mark.parametrize("far", [-0.001, 1.001, np.nan, np.inf, -np.inf, True, "0.1", [0.1]])
def test_invalid_far_raises(far):
    with pytest.raises(ValueError, match="far"):
        tar_at_far([1, 0], [0.8, 0.2], far)


@pytest.mark.parametrize("threshold", [np.nan, True, "0.5", [0.5]])
def test_invalid_accuracy_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        accuracy_at([1, 0], [0.8, 0.2], threshold)


@pytest.mark.parametrize("n_diff,far", [(300, 0.001), (999, 0.001), (99, 0.01)])
def test_positive_far_below_empirical_resolution_is_nan(n_diff, far):
    labels, scores = _inputs([0.9], np.linspace(0.0, 0.2, n_diff))
    assert np.isnan(tar_at_far(labels, scores, far)).all()


def test_far_minimum_resolution_and_endpoints():
    labels, scores = _inputs([0.9, 0.8], np.linspace(0.0, 0.3, 1000))
    tar, threshold, actual = tar_at_far(labels, scores, 0.001)
    assert tar == 1.0
    assert actual == 0.001
    assert threshold == scores[-2]
    assert tar_at_far(labels, scores, 0.0) == (1.0, 0.3, 0.0)
    assert tar_at_far(labels, scores, 1.0) == (1.0, -np.inf, 1.0)


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_tar_empty_or_missing_class_is_nan(labels):
    assert np.isnan(tar_at_far(labels, np.zeros(len(labels)), 0.001)).all()


def test_empty_threshold_and_accuracy():
    assert np.isnan(best_threshold([], [])).all()
    assert np.isnan(accuracy_at([], [], 0.5))


def test_all_constant_scores_and_strict_comparison():
    labels, scores = _inputs(np.full(2, 0.5), np.full(1000, 0.5))
    assert tar_at_far(labels, scores, 0.001) == (0.0, 0.5, 0.0)
    assert accuracy_at([1, 0], [0.5, 0.5], 0.5) == 0.5


def test_ties_cannot_be_split_to_fill_false_accept_budget():
    labels, scores = _inputs([0.95, 0.8, 0.79], [0.9, 0.8, 0.8, 0.1, 0.0])
    assert tar_at_far(labels, scores, 0.4) == (pytest.approx(1 / 3), 0.8, 0.2)


def test_tar_threshold_does_not_depend_on_positive_scores():
    diff = np.linspace(-0.2, 0.3, 1000)
    first = tar_at_far(*_inputs([0.9], diff), 0.01)
    second = tar_at_far(*_inputs([-0.3] * 50, diff), 0.01)
    assert first[1:] == second[1:]
    assert first[0] == 1.0 and second[0] == 0.0


@pytest.mark.parametrize("labels", [[1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]])
def test_best_threshold_constant_scores_selects_majority(labels):
    scores = np.full(len(labels), 0.4)
    threshold, accuracy = best_threshold(labels, scores)
    expected = max(np.mean(np.array(labels) == 1), np.mean(np.array(labels) == 0))
    assert accuracy == expected
    assert accuracy_at(labels, scores, threshold) == expected


@pytest.mark.parametrize("scores", [
    [np.nextafter(0.5, -np.inf), 0.5],
    [1e308, np.nextafter(1e308, np.inf)],
    [-np.finfo(float).max, np.finfo(float).max],
])
def test_best_threshold_separates_adjacent_and_extreme_floats(scores):
    threshold, accuracy = best_threshold([0, 1], scores)
    assert accuracy == 1.0
    assert accuracy_at([0, 1], scores, threshold) == 1.0


def test_best_threshold_matches_exhaustive_decision_oracle():
    rng = np.random.default_rng(26)
    for _ in range(60):
        labels = rng.integers(0, 2, size=40)
        scores = rng.integers(-3, 4, size=40) / 4.0
        # With strict >, the observed scores plus a value below the minimum
        # enumerate every possible decision boundary; this is independent of midpoints.
        candidates = np.r_[scores.min() - 1.0, np.unique(scores)]
        expected = max(np.mean((scores > t) == (labels == 1)) for t in candidates)
        threshold, accuracy = best_threshold(labels, scores)
        assert accuracy == expected
        assert accuracy_at(labels, scores, threshold) == expected


def test_tar_permutation_and_no_mutation():
    labels, scores = _inputs([0.9, 0.8], np.linspace(-0.3, 0.3, 1000))
    labels_before, scores_before = labels.copy(), scores.copy()
    expected = tar_at_far(labels, scores, 0.001)
    order = np.random.default_rng(26).permutation(len(labels))
    assert tar_at_far(labels[order], scores[order], 0.001) == expected
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(scores, scores_before)


@pytest.mark.parametrize("n_diff,expected_status", [(300, "insufficient_far_resolution"), (1000, "ok"), (0, "missing_class")])
def test_summary_reports_resolution_status(n_diff, expected_status):
    labels, scores = _inputs([0.8, 0.9], np.full(n_diff, 0.2))
    sc = Scores(labels, scores, np.ones(len(labels), dtype=int))
    result = summarize(sc, far_targets=(0.001,))["tar_at_far"]["1e-03"]
    assert result["status"] == expected_status
    assert result["n_impostor"] == n_diff
    assert result["far_resolution"] == (1 / n_diff if n_diff else None)
    if expected_status != "ok":
        assert np.isnan(result["tar"])
        assert np.isnan(result["threshold"])
        assert np.isnan(result["actual_far"])
    else:
        assert result["tar"] == 1.0
