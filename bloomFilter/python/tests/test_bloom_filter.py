from math import log, ceil

import pytest

from bloom_filter import BloomFilter
from bloom_filter.bloom_filter import BIT_ALIGNMENT

# ---------------------------------------------------------------------------
# Construction / Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_max_size(self):
        bf = BloomFilter(100, 0.01)
        assert bf.max_elements == 100

    def test_stores_false_positive_rate(self):
        bf = BloomFilter(100, 0.01)
        assert bf.false_positive_rate == 0.01

    def test_bit_array_starts_all_zeros(self):
        bf = BloomFilter(100, 0.01)
        assert bf.bit_array.count(1) == 0

    def test_bit_array_size_is_64_bit_aligned(self):
        for n in [1, 7, 50, 999, 10_000]:
            bf = BloomFilter(n, 0.05)
            assert bf.bit_array_size % BIT_ALIGNMENT == 0

    def test_hash_count_is_positive(self):
        bf = BloomFilter(100, 0.01)
        assert bf.hash_count >= 1

    def test_larger_capacity_produces_larger_bit_array(self):
        small = BloomFilter(100, 0.01)
        large = BloomFilter(10_000, 0.01)
        assert large.bit_array_size > small.bit_array_size

    def test_lower_fp_rate_produces_larger_bit_array(self):
        relaxed = BloomFilter(1000, 0.1)
        strict = BloomFilter(1000, 0.001)
        assert strict.bit_array_size > relaxed.bit_array_size

    def test_single_element_capacity(self):
        bf = BloomFilter(1, 0.01)
        assert bf.bit_array_size >= 1
        assert bf.hash_count >= 1


# ---------------------------------------------------------------------------
# Static helper methods
# ---------------------------------------------------------------------------


class TestCalculateBitSize:
    def test_known_formula(self):
        n, p = 1000, 0.01
        expected = int(ceil(-(n * log(p) / log(2) ** 2)))
        assert BloomFilter.calculate_bit_size(n, p) == expected

    def test_returns_positive_for_valid_inputs(self):
        assert BloomFilter.calculate_bit_size(10, 0.5) > 0

    def test_fp_rate_near_one_returns_small_size(self):
        m = BloomFilter.calculate_bit_size(100, 0.99)
        assert m >= 1

    def test_very_small_fp_rate(self):
        m = BloomFilter.calculate_bit_size(100, 1e-10)
        assert m > BloomFilter.calculate_bit_size(100, 0.01)


class TestCalculateHashCount:
    def test_known_formula(self):
        n, m = 1000, 9586
        expected = int(ceil((m / n) * log(2)))
        assert BloomFilter.calculate_hash_count(n, m) == expected

    def test_returns_at_least_one(self):
        assert BloomFilter.calculate_hash_count(1000, 10) >= 1


# ---------------------------------------------------------------------------
# add() and contains() — core behavior
# ---------------------------------------------------------------------------


class TestAddAndContains:
    def test_contains_returns_false_on_empty_filter(self):
        bf = BloomFilter(100, 0.01)
        assert bf.contains("anything") is False

    def test_added_item_is_found(self):
        bf = BloomFilter(100, 0.01)
        bf.add("hello")
        assert bf.contains("hello") is True

    def test_multiple_items_all_found(self):
        bf = BloomFilter(1000, 0.01)
        items = [f"item-{i}" for i in range(100)]
        for item in items:
            bf.add(item)
        for item in items:
            assert bf.contains(item) is True, f"{item} not found"

    def test_unadded_item_not_found(self):
        bf = BloomFilter(100, 0.01)
        bf.add("present")
        assert bf.contains("absent") is False

    def test_adding_same_item_twice_is_idempotent(self):
        bf = BloomFilter(100, 0.01)
        bf.add("dup")
        bits_after_first = bf.bit_array.copy()
        bf.add("dup")
        assert bf.bit_array == bits_after_first

    def test_add_sets_bits_in_array(self):
        bf = BloomFilter(100, 0.01)
        bf.add("test")
        assert bf.bit_array.count(1) >= 1


# ---------------------------------------------------------------------------
# Edge-case inputs to add / contains
# ---------------------------------------------------------------------------


class TestEdgeCaseInputs:
    def test_empty_string(self):
        bf = BloomFilter(100, 0.01)
        bf.add("")
        assert bf.contains("") is True
        assert bf.contains("notempty") is False

    def test_very_long_string(self):
        bf = BloomFilter(100, 0.01)
        long_str = "a" * 100_000
        bf.add(long_str)
        assert bf.contains(long_str) is True

    def test_unicode_strings(self):
        bf = BloomFilter(100, 0.01)
        bf.add("cafe\u0301")
        assert bf.contains("cafe\u0301") is True

    def test_emoji(self):
        bf = BloomFilter(100, 0.01)
        bf.add("\U0001f600\U0001f680")
        assert bf.contains("\U0001f600\U0001f680") is True

    def test_special_characters(self):
        bf = BloomFilter(100, 0.01)
        specials = "!@#$%^&*()_+-=[]{}|;':\",./<>?\\\n\t\0"
        bf.add(specials)
        assert bf.contains(specials) is True

    def test_whitespace_only(self):
        bf = BloomFilter(100, 0.01)
        bf.add("   ")
        assert bf.contains("   ") is True
        assert bf.contains("  ") is False

    def test_similar_strings_distinguished(self):
        bf = BloomFilter(1000, 0.001)
        bf.add("abc")
        assert bf.contains("abc") is True
        assert bf.contains("abd") is False
        assert bf.contains("ab") is False
        assert bf.contains("abcd") is False


# ---------------------------------------------------------------------------
# False positive rate — statistical validation
# ---------------------------------------------------------------------------


class TestFalsePositiveRate:
    def test_empirical_fp_rate_within_tolerance(self):
        n = 10_000
        target_fp = 0.05
        bf = BloomFilter(n, target_fp)

        for i in range(n):
            bf.add(f"member-{i}")

        trials = 50_000
        false_positives = sum(1 for i in range(trials) if bf.contains(f"nonmember-{i}"))
        observed_fp = false_positives / trials

        assert (
            observed_fp < target_fp * 2
        ), f"Observed FP rate {observed_fp:.4f} exceeds 2x target {target_fp}"

    def test_no_false_negatives(self):
        bf = BloomFilter(5000, 0.01)
        items = [f"item-{i}" for i in range(5000)]
        for item in items:
            bf.add(item)
        for item in items:
            assert bf.contains(item) is True


# ---------------------------------------------------------------------------
# Combined hash function
# ---------------------------------------------------------------------------


class TestCalculateCombinedHash:
    def test_hash_within_bounds(self):
        bf = BloomFilter(100, 0.01)
        for i in range(bf.hash_count):
            h = bf._calculate_combined_hash("test", i)
            assert 0 <= h < bf.bit_array_size

    def test_different_indices_produce_different_hashes(self):
        bf = BloomFilter(1000, 0.01)
        hashes = [bf._calculate_combined_hash("test", i) for i in range(bf.hash_count)]
        assert len(set(hashes)) > 1

    def test_different_items_produce_different_hashes(self):
        bf = BloomFilter(1000, 0.01)
        h1 = bf._calculate_combined_hash("alpha", 0)
        h2 = bf._calculate_combined_hash("beta", 0)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Invalid / boundary constructor arguments
# ---------------------------------------------------------------------------


class TestInvalidConstructorArgs:
    def test_zero_max_elements(self):
        with pytest.raises(ValueError):
            BloomFilter(0, 0.01)

    def test_negative_max_elements(self):
        with pytest.raises(ValueError):
            BloomFilter(-1, 0.01)

    def test_fp_rate_zero(self):
        with pytest.raises(ValueError):
            BloomFilter(100, 0.0)

    def test_fp_rate_one(self):
        with pytest.raises(ValueError):
            BloomFilter(100, 1.0)

    def test_fp_rate_greater_than_one(self):
        with pytest.raises(ValueError):
            BloomFilter(100, 1.5)

    def test_fp_rate_negative(self):
        with pytest.raises(ValueError):
            BloomFilter(100, -0.01)


# ---------------------------------------------------------------------------
# __contains__ (in operator) support
# ---------------------------------------------------------------------------


class TestContainsOperator:
    def test_in_operator_returns_false_on_empty_filter(self):
        bf = BloomFilter(100, 0.01)
        assert ("anything" in bf) is False

    def test_in_operator_finds_added_item(self):
        bf = BloomFilter(100, 0.01)
        bf.add("hello")
        assert "hello" in bf

    def test_in_operator_does_not_find_unadded_item(self):
        bf = BloomFilter(100, 0.01)
        bf.add("present")
        assert ("absent" in bf) is False

    def test_in_operator_multiple_items(self):
        bf = BloomFilter(1000, 0.01)
        items = [f"item-{i}" for i in range(50)]
        for item in items:
            bf.add(item)
        for item in items:
            assert item in bf

    def test_in_operator_equivalent_to_contains(self):
        bf = BloomFilter(500, 0.01)
        test_items = ["alpha", "beta", "gamma", "delta"]
        bf.add("alpha")
        bf.add("gamma")

        for item in test_items:
            assert (item in bf) == bf.contains(item)

    def test_in_operator_with_empty_string(self):
        bf = BloomFilter(100, 0.01)
        bf.add("")
        assert "" in bf

    def test_in_operator_with_unicode(self):
        bf = BloomFilter(100, 0.01)
        bf.add("café")
        assert "café" in bf
        assert "cafe" not in bf


# ---------------------------------------------------------------------------
# __repr__ string representation
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_class_name(self):
        bf = BloomFilter(100, 0.01)
        assert "BloomFilter" in repr(bf)

    def test_repr_contains_max_elements(self):
        bf = BloomFilter(100, 0.01)
        assert "max_elements=100" in repr(bf)

    def test_repr_contains_false_positive_rate(self):
        bf = BloomFilter(100, 0.01)
        r = repr(bf)
        assert "false_positive_rate=" in r
        assert "0.01" in r

    def test_repr_contains_bit_array_size(self):
        bf = BloomFilter(100, 0.01)
        r = repr(bf)
        assert "bit_array_size=" in r
        assert str(bf.bit_array_size) in r

    def test_repr_contains_hash_count(self):
        bf = BloomFilter(100, 0.01)
        r = repr(bf)
        assert "hash_count=" in r
        assert str(bf.hash_count) in r

    def test_repr_contains_bitarray(self):
        bf = BloomFilter(100, 0.01)
        assert "bitarray=" in repr(bf)

    def test_repr_fp_rate_formatted_with_six_decimals(self):
        bf = BloomFilter(100, 0.123456789)
        r = repr(bf)
        assert "false_positive_rate=0.123457" in r

    def test_repr_different_for_different_filters(self):
        bf1 = BloomFilter(100, 0.01)
        bf2 = BloomFilter(200, 0.05)
        assert repr(bf1) != repr(bf2)

    def test_repr_reflects_added_items(self):
        bf = BloomFilter(100, 0.01)
        repr_before = repr(bf)
        bf.add("test")
        repr_after = repr(bf)
        assert repr_before != repr_after

    def test_repr_includes_all_parameters(self):
        bf = BloomFilter(500, 0.001)
        r = repr(bf)
        assert "max_elements=500" in r
        assert "false_positive_rate=" in r
        assert "bit_array_size=" in r
        assert "hash_count=" in r
        assert "bitarray=" in r
