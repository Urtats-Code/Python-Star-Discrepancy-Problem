import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
from PermutationHashMap import PermutationHashMap


@pytest.fixture
def hashmap():
    return PermutationHashMap()


# --- Normal behaviour ---

def test_key_from_list(hashmap):
    assert hashmap._key([1, 2, 3]) == (1, 2, 3)

def test_key_from_numpy_array(hashmap):
    perm = np.array([3, 1, 2])
    assert hashmap._key(perm) == (3, 1, 2)

def test_key_returns_tuple(hashmap):
    result = hashmap._key([2, 1, 3])
    assert isinstance(result, tuple)

def test_key_same_permutation_same_key(hashmap):
    assert hashmap._key([1, 2, 3]) == hashmap._key([1, 2, 3])

def test_key_different_permutations_different_keys(hashmap):
    assert hashmap._key([1, 2, 3]) != hashmap._key([3, 2, 1])


# --- None value ---

def test_key_none_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap._key(None)

def test_key_list_containing_none_raises(hashmap):
    with pytest.raises((TypeError, ValueError)):
        hashmap._key([1, None, 3])


# --- Extremely large permutation ---

def test_key_large_permutation(hashmap):
    perm = list(range(1, 10_001))
    result = hashmap._key(perm)
    assert isinstance(result, tuple)
    assert len(result) == 10_000
    assert result[0] == 1
    assert result[-1] == 10_000

def test_key_large_numpy_permutation(hashmap):
    perm = np.arange(1, 10_001)
    result = hashmap._key(perm)
    assert len(result) == 10_000

def test_key_large_integers(hashmap):
    perm = [10**18, 10**18 + 1, 10**18 + 2]
    result = hashmap._key(perm)
    assert result == (10**18, 10**18 + 1, 10**18 + 2)


# --- Non-valid types as key ---

def test_key_string_of_digits_raises(hashmap):
    # A plain string is iterable, yielding characters — int('1') works but
    # int('12') from a single char won't; more importantly strings are not
    # valid permutation inputs and should raise.
    with pytest.raises((ValueError, TypeError)):
        hashmap._key("abc")

def test_key_integer_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap._key(42)

def test_key_float_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap._key(3.14)

def test_key_dict_iterates_keys(hashmap):
    # Iterating a dict yields its keys; if keys are ints, _key succeeds silently.
    result = hashmap._key({1: "a", 2: "b"})
    assert result == (1, 2)

def test_key_list_of_floats_truncates_to_int(hashmap):
    # float -> int conversion is defined behaviour in _key
    result = hashmap._key([1.9, 2.1, 3.7])
    assert result == (1, 2, 3)

def test_key_list_of_strings_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap._key(["a", "b", "c"])


# ===========================================================================
# set() tests
# ===========================================================================

# --- Normal behaviour ---

def test_set_stores_value(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.get([1, 2, 3]) == 0.5

def test_set_with_numpy_permutation(hashmap):
    perm = np.array([3, 1, 2])
    hashmap.set(perm, 1.23)
    assert hashmap.get(perm) == 1.23

def test_set_overwrites_existing_value(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    hashmap.set([1, 2, 3], 9.9)
    assert hashmap.get([1, 2, 3]) == 9.9

def test_set_different_permutations_stored_independently(hashmap):
    hashmap.set([1, 2, 3], 1.0)
    hashmap.set([3, 2, 1], 2.0)
    assert hashmap.get([1, 2, 3]) == 1.0
    assert hashmap.get([3, 2, 1]) == 2.0

def test_set_increases_length(hashmap):
    hashmap.set([1, 2, 3], 0.1)
    hashmap.set([2, 3, 1], 0.2)
    assert len(hashmap) == 2

def test_set_integer_value_converted_to_float(hashmap):
    hashmap.set([1, 2, 3], 5)
    result = hashmap.get([1, 2, 3])
    assert isinstance(result, float)
    assert result == 5.0


# --- Wrong type for value ---

def test_set_string_value_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap.set([1, 2, 3], "bad")

def test_set_none_value_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap.set([1, 2, 3], None)

def test_set_list_value_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap.set([1, 2, 3], [0.5])

def test_set_invalid_permutation_type_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.set(42, 0.5)


# --- None values ---

def test_set_none_permutation_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.set(None, 0.5)

def test_set_permutation_containing_none_raises(hashmap):
    with pytest.raises((TypeError, ValueError)):
        hashmap.set([1, None, 3], 0.5)


# --- Extremely large values ---

def test_set_large_permutation(hashmap):
    perm = list(range(1, 10_001))
    hashmap.set(perm, 42.0)
    assert hashmap.get(perm) == 42.0

def test_set_large_float_value(hashmap):
    hashmap.set([1, 2, 3], 1.7976931348623157e+308)
    assert hashmap.get([1, 2, 3]) == 1.7976931348623157e+308

def test_set_very_small_float_value(hashmap):
    hashmap.set([1, 2, 3], 5e-324)
    assert hashmap.get([1, 2, 3]) == 5e-324

def test_set_large_integer_elements(hashmap):
    perm = [10**18, 10**18 + 1, 10**18 + 2]
    hashmap.set(perm, 0.99)
    assert hashmap.get(perm) == 0.99


# ===========================================================================
# get() tests
# ===========================================================================

# --- Normal behaviour ---

def test_get_returns_stored_value(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.get([1, 2, 3]) == 0.5

def test_get_with_numpy_permutation(hashmap):
    perm = np.array([2, 3, 1])
    hashmap.set(perm, 7.7)
    assert hashmap.get(perm) == 7.7

def test_get_list_and_numpy_equivalent(hashmap):
    hashmap.set([1, 2, 3], 3.14)
    assert hashmap.get(np.array([1, 2, 3])) == 3.14

def test_get_returns_latest_after_overwrite(hashmap):
    hashmap.set([1, 2, 3], 1.0)
    hashmap.set([1, 2, 3], 2.0)
    assert hashmap.get([1, 2, 3]) == 2.0

def test_get_returns_float(hashmap):
    hashmap.set([1, 2, 3], 4)
    assert isinstance(hashmap.get([1, 2, 3]), float)


# --- Non-existent keys ---

def test_get_missing_key_returns_none(hashmap):
    assert hashmap.get([1, 2, 3]) is None

def test_get_different_order_returns_none(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.get([3, 2, 1]) is None

def test_get_on_empty_hashmap_returns_none(hashmap):
    assert hashmap.get([1, 2, 3]) is None

def test_get_after_set_other_key_returns_none(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.get([2, 3, 1]) is None


# --- Wrong value types as permutation ---

def test_get_none_permutation_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.get(None)

def test_get_integer_permutation_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.get(42)

def test_get_float_permutation_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.get(3.14)

def test_get_string_permutation_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap.get("abc")

def test_get_list_containing_none_raises(hashmap):
    with pytest.raises((TypeError, ValueError)):
        hashmap.get([1, None, 3])

def test_get_list_of_strings_raises(hashmap):
    with pytest.raises((ValueError, TypeError)):
        hashmap.get(["a", "b", "c"])


# --- Extremely large values ---

def test_get_large_permutation(hashmap):
    perm = list(range(1, 10_001))
    hashmap.set(perm, 99.0)
    assert hashmap.get(perm) == 99.0

def test_get_large_permutation_missing_returns_none(hashmap):
    perm = list(range(1, 10_001))
    assert hashmap.get(perm) is None

def test_get_large_float_value(hashmap):
    hashmap.set([1, 2, 3], 1.7976931348623157e+308)
    assert hashmap.get([1, 2, 3]) == 1.7976931348623157e+308

def test_get_very_small_float_value(hashmap):
    hashmap.set([1, 2, 3], 5e-324)
    assert hashmap.get([1, 2, 3]) == 5e-324

def test_get_large_integer_elements(hashmap):
    perm = [10**18, 10**18 + 1, 10**18 + 2]
    hashmap.set(perm, 0.42)
    assert hashmap.get(perm) == 0.42


# ===========================================================================
# contains() tests
# ===========================================================================

def test_contains_existing_key(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.contains([1, 2, 3]) is True

def test_contains_missing_key(hashmap):
    assert hashmap.contains([1, 2, 3]) is False

def test_contains_different_order_is_false(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.contains([3, 2, 1]) is False

def test_contains_numpy_equivalent(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert hashmap.contains(np.array([1, 2, 3])) is True

def test_contains_none_raises(hashmap):
    with pytest.raises(TypeError):
        hashmap.contains(None)


# ===========================================================================
# __len__() tests
# ===========================================================================

def test_len_empty(hashmap):
    assert len(hashmap) == 0

def test_len_after_single_set(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    assert len(hashmap) == 1

def test_len_after_multiple_sets(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    hashmap.set([2, 3, 1], 1.0)
    hashmap.set([3, 1, 2], 1.5)
    assert len(hashmap) == 3

def test_len_overwrite_does_not_increase(hashmap):
    hashmap.set([1, 2, 3], 0.5)
    hashmap.set([1, 2, 3], 9.9)
    assert len(hashmap) == 1
