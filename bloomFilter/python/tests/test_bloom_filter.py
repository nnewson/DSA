from math import log, ceil

import pytest

from bloom_filter import BloomFilter

BYTE_ALIGNMENT = 8

# ---------------------------------------------------------------------------
# Construction / Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_max_size(self):
        bf = BloomFilter(100, 0.01)
        assert bf._max_elements == 100

    def test_stores_false_positive_rate(self):
        bf = BloomFilter(100, 0.01)
        assert bf._false_positive_rate == 0.01

    def test_bit_array_starts_all_zeros(self):
        bf = BloomFilter(100, 0.01)
        assert bf._bit_array.count(1) == 0

    def test_bit_array_size_is_64_bit_aligned(self):
        for n in [1, 7, 50, 999, 10_000]:
            bf = BloomFilter(n, 0.05)
            assert bf._bit_array_size % BYTE_ALIGNMENT == 0

    def test_hash_count_is_positive(self):
        bf = BloomFilter(100, 0.01)
        assert bf._hash_count >= 1

    def test_larger_capacity_produces_larger_bit_array(self):
        small = BloomFilter(100, 0.01)
        large = BloomFilter(10_000, 0.01)
        assert large._bit_array_size > small._bit_array_size

    def test_lower_fp_rate_produces_larger_bit_array(self):
        relaxed = BloomFilter(1000, 0.1)
        strict = BloomFilter(1000, 0.001)
        assert strict._bit_array_size > relaxed._bit_array_size

    def test_single_element_capacity(self):
        bf = BloomFilter(1, 0.01)
        assert bf._bit_array_size >= 1
        assert bf._hash_count >= 1


# ---------------------------------------------------------------------------
# Static helper methods
# ---------------------------------------------------------------------------


class TestCalculateBitSize:
    def test_known_formula(self):
        n, p = 1000, 0.01
        expected = int(ceil(-(n * log(p) / log(2) ** 2)))
        assert BloomFilter._calculate_bit_size(n, p) == expected

    def test_returns_positive_for_valid_inputs(self):
        assert BloomFilter._calculate_bit_size(10, 0.5) > 0

    def test_fp_rate_near_one_returns_small_size(self):
        m = BloomFilter._calculate_bit_size(100, 0.99)
        assert m >= 1

    def test_very_small_fp_rate(self):
        m = BloomFilter._calculate_bit_size(100, 1e-10)
        assert m > BloomFilter._calculate_bit_size(100, 0.01)


class TestCalculateHashCount:
    def test_known_formula(self):
        n, m = 1000, 9586
        expected = int(ceil((m / n) * log(2)))
        assert BloomFilter._calculate_hash_count(n, m) == expected

    def test_returns_at_least_one(self):
        assert BloomFilter._calculate_hash_count(1000, 10) >= 1


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
        bits_after_first = bf._bit_array.copy()
        bf.add("dup")
        assert bf._bit_array == bits_after_first

    def test_add_sets_bits_in_array(self):
        bf = BloomFilter(100, 0.01)
        bf.add("test")
        assert bf._bit_array.count(1) >= 1


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
        for i in range(bf._hash_count):
            h = bf._calculate_combined_hash("test", i)
            assert 0 <= h < bf._bit_array_size

    def test_different_indices_produce_different_hashes(self):
        bf = BloomFilter(1000, 0.01)
        hashes = [bf._calculate_combined_hash("test", i) for i in range(bf._hash_count)]
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
