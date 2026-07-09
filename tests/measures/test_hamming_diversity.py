import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
from measures.HammingDiversity import HammingDiversity


@pytest.fixture
def tracker():
    return HammingDiversity()


# --- pairwise_distances() ---

def test_pairwise_distances_identical_population_is_zero():
    population = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    distances = HammingDiversity.pairwise_distances(population)
    assert np.all(distances == 0)

def test_pairwise_distances_fully_reversed_pair():
    population = [[1, 2, 3], [3, 2, 1]]
    distances = HammingDiversity.pairwise_distances(population)
    assert list(distances) == [2]

def test_pairwise_distances_count_matches_number_of_pairs():
    population = [[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2]]
    distances = HammingDiversity.pairwise_distances(population)
    assert len(distances) == 6  # C(4, 2)

def test_pairwise_distances_single_permutation_is_empty():
    distances = HammingDiversity.pairwise_distances([[1, 2, 3]])
    assert distances.size == 0

def test_pairwise_distances_empty_population_is_empty():
    distances = HammingDiversity.pairwise_distances(np.empty((0, 3), dtype=int))
    assert distances.size == 0


# --- summary() ---

def test_summary_identical_population():
    population = [[1, 2, 3], [1, 2, 3]]
    result = HammingDiversity.summary(population)
    assert result["mean"] == 0.0
    assert result["std"] == 0.0
    assert result["min"] == 0.0
    assert result["max"] == 0.0
    assert result["mean_normalized"] == 0.0

def test_summary_maximally_different_pair():
    population = [[1, 2, 3], [3, 1, 2]]
    result = HammingDiversity.summary(population)
    assert result["mean"] == 3.0
    assert result["mean_normalized"] == 1.0

def test_summary_mean_normalized_is_within_bounds():
    population = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], [2, 1, 4, 3]]
    result = HammingDiversity.summary(population)
    assert 0.0 <= result["mean_normalized"] <= 1.0

def test_summary_single_permutation_returns_zeros():
    result = HammingDiversity.summary([[1, 2, 3]])
    assert result == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "mean_normalized": 0.0}

def test_summary_accepts_numpy_array():
    population = np.array([[1, 2, 3], [3, 2, 1]])
    result = HammingDiversity.summary(population)
    assert result["mean"] == 2.0

def test_summary_min_max_reflect_extremes():
    population = [[1, 2, 3], [1, 2, 3], [3, 2, 1]]
    result = HammingDiversity.summary(population)
    assert result["min"] == 0.0
    assert result["max"] == 2.0


# --- record() / history ---

def test_record_appends_to_history(tracker):
    tracker.record(0, [[1, 2, 3], [3, 2, 1]])
    assert len(tracker.history) == 1

def test_record_includes_generation_number(tracker):
    entry = tracker.record(5, [[1, 2, 3], [3, 2, 1]])
    assert entry["generation"] == 5

def test_record_returns_summary_fields(tracker):
    entry = tracker.record(0, [[1, 2, 3], [3, 2, 1]])
    assert entry["mean"] == 2.0

def test_record_accumulates_across_generations(tracker):
    tracker.record(0, [[1, 2, 3], [1, 2, 3]])
    tracker.record(1, [[1, 2, 3], [3, 2, 1]])
    assert [e["generation"] for e in tracker.history] == [0, 1]
    assert [e["mean"] for e in tracker.history] == [0.0, 2.0]

def test_new_tracker_has_empty_history(tracker):
    assert tracker.history == []
