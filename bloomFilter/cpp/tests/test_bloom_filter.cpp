#include <cmath>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "bloom_filter/bloom_filter.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void addString(bloomFilter& bf, const std::string& s) {
    bf.add(reinterpret_cast<const std::byte*>(s.data()), s.size());
}

static bool containsString(const bloomFilter& bf, const std::string& s) {
    return bf.contains(reinterpret_cast<const std::byte*>(s.data()), s.size());
}

static size_t combinedHashString(const bloomFilter& bf, const std::string& s, size_t i) {
    return bf.calculateCombinedHash(reinterpret_cast<const std::byte*>(s.data()), s.size(), i);
}

// ---------------------------------------------------------------------------
// Construction / Initialization
// ---------------------------------------------------------------------------

TEST(Init, StoresMaxElements) {
    bloomFilter bf(100, 0.01);
    EXPECT_EQ(bf.getMaxElements(), 100u);
}

TEST(Init, StoresFalsePositiveRate) {
    bloomFilter bf(100, 0.01);
    EXPECT_DOUBLE_EQ(bf.getFalsePositiveRate(), 0.01);
}

TEST(Init, BitArrayStartsAllZeros) {
    bloomFilter bf(100, 0.01);
    const auto& bits = bf.getBitArray();
    size_t count = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) count++;
    }
    EXPECT_EQ(count, 0u);
}

TEST(Init, BitArraySizeIs64BitAligned) {
    for (size_t n : {1, 7, 50, 999, 10000}) {
        bloomFilter bf(n, 0.05);
        EXPECT_EQ(bf.getBitArraySize() % BIT_ALIGNMENT, 0u) << "n=" << n;
    }
}

TEST(Init, HashCountIsPositive) {
    bloomFilter bf(100, 0.01);
    EXPECT_GE(bf.getHashCount(), 1u);
}

TEST(Init, LargerCapacityProducesLargerBitArray) {
    bloomFilter small(100, 0.01);
    bloomFilter large(10000, 0.01);
    EXPECT_GT(large.getBitArraySize(), small.getBitArraySize());
}

TEST(Init, LowerFpRateProducesLargerBitArray) {
    bloomFilter relaxed(1000, 0.1);
    bloomFilter strict(1000, 0.001);
    EXPECT_GT(strict.getBitArraySize(), relaxed.getBitArraySize());
}

TEST(Init, SingleElementCapacity) {
    bloomFilter bf(1, 0.01);
    EXPECT_GE(bf.getBitArraySize(), 1u);
    EXPECT_GE(bf.getHashCount(), 1u);
}

// ---------------------------------------------------------------------------
// Static helper methods
// ---------------------------------------------------------------------------

TEST(CalculateBitSize, KnownFormula) {
    size_t n = 1000;
    double p = 0.01;
    size_t expected = static_cast<size_t>(
        std::ceil(-(static_cast<double>(n) * std::log(p)) / std::pow(std::log(2.0), 2.0))
    );
    EXPECT_EQ(bloomFilter::calculateBitSize(n, p), expected);
}

TEST(CalculateBitSize, ReturnsPositiveForValidInputs) {
    EXPECT_GT(bloomFilter::calculateBitSize(10, 0.5), 0u);
}

TEST(CalculateBitSize, FpRateNearOneReturnsSmallSize) {
    size_t m = bloomFilter::calculateBitSize(100, 0.99);
    EXPECT_GE(m, 1u);
}

TEST(CalculateBitSize, VerySmallFpRate) {
    size_t m = bloomFilter::calculateBitSize(100, 1e-10);
    EXPECT_GT(m, bloomFilter::calculateBitSize(100, 0.01));
}

TEST(CalculateHashCount, KnownFormula) {
    size_t n = 1000;
    size_t m = 9586;
    size_t expected = static_cast<size_t>(
        std::ceil((static_cast<double>(m) / n) * std::log(2.0))
    );
    EXPECT_EQ(bloomFilter::calculateHashCount(n, m), expected);
}

TEST(CalculateHashCount, ReturnsAtLeastOne) {
    EXPECT_GE(bloomFilter::calculateHashCount(1000, 10), 1u);
}

// ---------------------------------------------------------------------------
// add() and contains() — core behavior
// ---------------------------------------------------------------------------

TEST(AddAndContains, ContainsReturnsFalseOnEmptyFilter) {
    bloomFilter bf(100, 0.01);
    EXPECT_FALSE(containsString(bf, "anything"));
}

TEST(AddAndContains, AddedItemIsFound) {
    bloomFilter bf(100, 0.01);
    addString(bf, "hello");
    EXPECT_TRUE(containsString(bf, "hello"));
}

TEST(AddAndContains, MultipleItemsAllFound) {
    bloomFilter bf(1000, 0.01);
    std::vector<std::string> items;
    for (int i = 0; i < 100; ++i) {
        items.push_back("item-" + std::to_string(i));
    }
    for (const auto& item : items) {
        addString(bf, item);
    }
    for (const auto& item : items) {
        EXPECT_TRUE(containsString(bf, item)) << item << " not found";
    }
}

TEST(AddAndContains, UnaddedItemNotFound) {
    bloomFilter bf(100, 0.01);
    addString(bf, "present");
    EXPECT_FALSE(containsString(bf, "absent"));
}

TEST(AddAndContains, AddingSameItemTwiceIsIdempotent) {
    bloomFilter bf(100, 0.01);
    addString(bf, "dup");
    const auto bitsAfterFirst = bf.getBitArray();
    addString(bf, "dup");
    EXPECT_EQ(bf.getBitArray(), bitsAfterFirst);
}

TEST(AddAndContains, AddSetsBitsInArray) {
    bloomFilter bf(100, 0.01);
    addString(bf, "test");
    const auto& bits = bf.getBitArray();
    size_t count = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) count++;
    }
    EXPECT_GE(count, 1u);
}

// ---------------------------------------------------------------------------
// Edge-case inputs to add / contains
// ---------------------------------------------------------------------------

TEST(EdgeCaseInputs, EmptyString) {
    bloomFilter bf(100, 0.01);
    addString(bf, "");
    EXPECT_TRUE(containsString(bf, ""));
    EXPECT_FALSE(containsString(bf, "notempty"));
}

TEST(EdgeCaseInputs, VeryLongString) {
    bloomFilter bf(100, 0.01);
    std::string longStr(100000, 'a');
    addString(bf, longStr);
    EXPECT_TRUE(containsString(bf, longStr));
}

TEST(EdgeCaseInputs, UnicodeStrings) {
    bloomFilter bf(100, 0.01);
    std::string s = "caf\xc3\xa9";  // "café" in UTF-8
    addString(bf, s);
    EXPECT_TRUE(containsString(bf, s));
}

TEST(EdgeCaseInputs, Emoji) {
    bloomFilter bf(100, 0.01);
    std::string s = "\xf0\x9f\x98\x80\xf0\x9f\x9a\x80";  // 😀🚀 in UTF-8
    addString(bf, s);
    EXPECT_TRUE(containsString(bf, s));
}

TEST(EdgeCaseInputs, SpecialCharacters) {
    bloomFilter bf(100, 0.01);
    std::string specials = "!@#$%^&*()_+-=[]{}|;':\",./<>?\\\n\t";
    specials += '\0';  // include null byte
    addString(bf, specials);
    EXPECT_TRUE(containsString(bf, specials));
}

TEST(EdgeCaseInputs, WhitespaceOnly) {
    bloomFilter bf(100, 0.01);
    addString(bf, "   ");
    EXPECT_TRUE(containsString(bf, "   "));
    EXPECT_FALSE(containsString(bf, "  "));
}

TEST(EdgeCaseInputs, SimilarStringsDistinguished) {
    bloomFilter bf(1000, 0.001);
    addString(bf, "abc");
    EXPECT_TRUE(containsString(bf, "abc"));
    EXPECT_FALSE(containsString(bf, "abd"));
    EXPECT_FALSE(containsString(bf, "ab"));
    EXPECT_FALSE(containsString(bf, "abcd"));
}

// ---------------------------------------------------------------------------
// False positive rate — statistical validation
// ---------------------------------------------------------------------------

TEST(FalsePositiveRate, EmpiricalFpRateWithinTolerance) {
    size_t n = 10000;
    double targetFp = 0.05;
    bloomFilter bf(n, targetFp);

    for (size_t i = 0; i < n; ++i) {
        addString(bf, "member-" + std::to_string(i));
    }

    size_t trials = 50000;
    size_t falsePositives = 0;
    for (size_t i = 0; i < trials; ++i) {
        if (containsString(bf, "nonmember-" + std::to_string(i))) {
            falsePositives++;
        }
    }
    double observedFp = static_cast<double>(falsePositives) / trials;

    EXPECT_LT(observedFp, targetFp * 2)
        << "Observed FP rate " << observedFp << " exceeds 2x target " << targetFp;
}

TEST(FalsePositiveRate, NoFalseNegatives) {
    bloomFilter bf(5000, 0.01);
    std::vector<std::string> items;
    for (int i = 0; i < 5000; ++i) {
        items.push_back("item-" + std::to_string(i));
    }
    for (const auto& item : items) {
        addString(bf, item);
    }
    for (const auto& item : items) {
        EXPECT_TRUE(containsString(bf, item));
    }
}

// ---------------------------------------------------------------------------
// Combined hash function
// ---------------------------------------------------------------------------

TEST(CombinedHash, HashWithinBounds) {
    bloomFilter bf(100, 0.01);
    for (size_t i = 1; i <= bf.getHashCount(); ++i) {
        size_t h = combinedHashString(bf, "test", i);
        EXPECT_LT(h, bf.getBitArraySize());
    }
}

TEST(CombinedHash, DifferentIndicesProduceDifferentHashes) {
    bloomFilter bf(1000, 0.01);
    std::set<size_t> hashes;
    for (size_t i = 1; i <= bf.getHashCount(); ++i) {
        hashes.insert(combinedHashString(bf, "test", i));
    }
    EXPECT_GT(hashes.size(), 1u);
}

TEST(CombinedHash, DifferentItemsProduceDifferentHashes) {
    bloomFilter bf(1000, 0.01);
    size_t h1 = combinedHashString(bf, "alpha", 1);
    size_t h2 = combinedHashString(bf, "beta", 1);
    EXPECT_NE(h1, h2);
}

// ---------------------------------------------------------------------------
// Invalid / boundary constructor arguments
// ---------------------------------------------------------------------------

TEST(InvalidConstructorArgs, ZeroMaxElements) {
    EXPECT_THROW(bloomFilter(0, 0.01), std::invalid_argument);
}

// Note: test_negative_max_elements is skipped because maxElements is size_t
// (unsigned), so a negative literal wraps to a large positive value.

TEST(InvalidConstructorArgs, FpRateZero) {
    EXPECT_THROW(bloomFilter(100, 0.0), std::invalid_argument);
}

TEST(InvalidConstructorArgs, FpRateOne) {
    EXPECT_THROW(bloomFilter(100, 1.0), std::invalid_argument);
}

TEST(InvalidConstructorArgs, FpRateGreaterThanOne) {
    EXPECT_THROW(bloomFilter(100, 1.5), std::invalid_argument);
}

TEST(InvalidConstructorArgs, FpRateNegative) {
    EXPECT_THROW(bloomFilter(100, -0.01), std::invalid_argument);
}
