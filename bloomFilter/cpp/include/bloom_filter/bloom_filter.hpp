#ifndef BLOOM_FILTER_HPP
#define BLOOM_FILTER_HPP    

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "xxhash.h"
#include "MurmurHash3.h"

namespace bloom_filter {

constexpr std::size_t BIT_ALIGNMENT = 64;
constexpr std::size_t BIT_ALIGNMENT_SHIFT = 6; // log2(64)
constexpr std::size_t BIT_ALIGNMENT_MASK = BIT_ALIGNMENT - 1;

class BitArray {
public:
    explicit BitArray(std::size_t bits)
        : words((bits + BIT_ALIGNMENT_MASK) / BIT_ALIGNMENT, 0) 
    {
    }

    void set(std::size_t i) 
    {
        words[i >> BIT_ALIGNMENT_SHIFT] |= (1ULL << (i & BIT_ALIGNMENT_MASK));
    }

    void clear(std::size_t i) 
    {
        words[i >> BIT_ALIGNMENT_SHIFT] &= ~(1ULL << (i & BIT_ALIGNMENT_MASK));
    }

    bool test(std::size_t i) const 
    {
        return words[i >> BIT_ALIGNMENT_SHIFT] & (1ULL << (i & BIT_ALIGNMENT_MASK));
    }

    std::size_t size() const 
    {
        return words.size() * BIT_ALIGNMENT;
    }

private:
    std::vector<std::uint64_t> words;
};

struct HashResult {
    std::size_t hash1;
    std::size_t hash2;
};

class BloomFilter {
public:
    /**
     * @brief Constructs a Bloom filter with specified capacity and false positive rate.
     * 
     * @param maxElements The maximum number of elements expected to be inserted into the filter.
     * @param falsePositiveRate The desired false positive probability (between 0 and 1).
     *                          Lower values require more memory but provide better accuracy.
     * 
     * @note The constructor automatically calculates the optimal number of hash functions
     *       and bit array size based on the provided parameters.
     */
    BloomFilter(std::size_t maxElements, double falsePositiveRate)
        : falsePositiveRate(falsePositiveRate),
          maxElements(maxElements),
          bitArray(calculateBitSize(maxElements, falsePositiveRate)),
          hashCount(calculateHashCount(maxElements, bitArray.size()))    
    {
    }

    /**
     * @brief Adds an element to the Bloom filter.
     * 
     * This method hashes the given element and sets the corresponding bits
     * in the Bloom filter's bit array. After this operation, future queries
     * for this element will return true (though false positives are possible).
     * 
     * @param element A span of constant bytes representing the element to be added.
     *                The span provides a view over the raw byte data without
     *                taking ownership.
     * 
     * @note This operation is irreversible - elements cannot be removed from
     *       a standard Bloom filter.
     * @note Thread safety depends on the implementation of the underlying bit array.
     */
    void add(std::span<const std::byte> element)
    {
        forEachPosition(element, [this](std::size_t position) 
        {
            this->bitArray.set(position);
            return true;
        });
    }

    [[nodiscard]]
    /**
     * @brief Checks if an element might be present in the Bloom filter.
     * 
     * @param element A span of bytes representing the element to check for membership.
     * @return true if the element might be in the set (possible false positive).
     * @return false if the element is definitely not in the set (no false negatives).
     * 
     * @note Due to the probabilistic nature of Bloom filters, a true result indicates
     *       the element was probably added, while a false result guarantees the element
     *       was never added.
     */
    bool contains(std::span<const std::byte> element) const
    {
        return forEachPosition(element, [this](std::size_t position) 
        {
            return this->bitArray.test(position);
        });
    }

    /**
     * @brief Gets the maximum number of elements the Bloom filter is designed to hold.
     * 
     * This method returns the maximum capacity of the Bloom filter that was specified
     * during construction. This value is used along with the desired false positive
     * probability to calculate the optimal filter size and number of hash functions.
     * 
     * @return The maximum number of elements the filter can efficiently store.
     * @note This function is marked noexcept and will not throw exceptions.
     * @note The [[nodiscard]] attribute indicates that ignoring the return value
     *       is likely an error.
     */
    [[nodiscard]]
    std::size_t getMaxElements() const noexcept
    {
        return maxElements; 
    }

    /**
     * @brief Gets the configured false positive rate of the Bloom filter.
     * 
     * Returns the theoretical false positive probability that was used to 
     * configure this Bloom filter. This is the expected probability of 
     * incorrectly reporting that an element is in the set when it is not.
     * 
     * @return The false positive rate as a double value between 0.0 and 1.0.
     * @note This function is noexcept and marked [[nodiscard]] to ensure the
     *       return value is not ignored.
     */
    [[nodiscard]]
    double getFalsePositiveRate() const noexcept
    { 
        return falsePositiveRate; 
    }

    /**
     * @brief Gets the size of the internal bit array.
     * 
     * @return The number of bits in the bit array.
     * @note This function is marked noexcept and [[nodiscard]], meaning it cannot throw exceptions
     *       and its return value should not be ignored.
     */
    [[nodiscard]]
    std::size_t getBitArraySize() const noexcept
    { 
        return bitArray.size(); 
    }

    /**
     * @brief Gets the number of hash functions used by the Bloom filter.
     * 
     * @return The number of hash functions used for element insertion and lookup.
     * 
     * @note This function is marked noexcept and does not throw exceptions.
     * @note The [[nodiscard]] attribute indicates the return value should not be ignored.
     */
    [[nodiscard]]
    std::size_t getHashCount() const noexcept
    { 
        return hashCount; 
    }

    /**
     * @brief Gets a constant reference to the internal bit array.
     * 
     * This method provides read-only access to the underlying bit array used by the bloom filter.
     * The bit array represents the state of the filter, where set bits indicate potential presence
     * of elements.
     * 
     * @return const BitArray& A constant reference to the internal bit array
     * @note This method is marked noexcept as it guarantees no exceptions will be thrown
     * @note The [[nodiscard]] attribute indicates that the return value should not be ignored
     */
    [[nodiscard]]
    const BitArray& getBitArray() const noexcept
    { 
        return bitArray; 
    }

    /**
     * @brief Calculates two 64-bit hash values for a given byte sequence.
     * 
     * This function computes two independent hash values using different hashing algorithms:
     * - XXH64 (xxHash) for the first hash value
     * - MurmurHash3 (x64_128 variant) for the second hash value
     * 
     * The second hash is constructed by combining the first two 32-bit values from the
     * MurmurHash3 output into a single 64-bit value.
     * 
     * @param element A span of bytes representing the element to be hashed
     * @return HashResult A structure containing two 64-bit hash values (hash1 and hash2)
     * 
     * @note This function is marked [[nodiscard]] to ensure the return value is not ignored
     * @note Both hash functions use a seed value of 0, as we're using double hashing 
     */
    [[nodiscard]]
    static HashResult calculateHashes(std::span<const std::byte> element) 
    {
        uint64_t hash1 = XXH64(element.data(), element.size(), 0);

        uint32_t mmh3Hash[4];
        MurmurHash3_x64_128(element.data(), element.size(), 0, mmh3Hash);
        uint64_t hash2 = (static_cast<uint64_t>(mmh3Hash[0]) << 32) | mmh3Hash[1];

        return {hash1, hash2};
    }

    /**
     * @brief Calculates the optimal bit array size for a Bloom filter.
     * 
     * This function computes the required number of bits for a Bloom filter
     * based on the desired maximum number of elements and target false positive rate.
     * The calculation uses the formula: m = -(n * ln(p)) / (ln(2))^2
     * where m is the bit array size, n is the number of elements, and p is the false positive rate.
     * 
     * @param maxElements The maximum number of elements to be stored in the Bloom filter.
     *                    Must be greater than 0.
     * @param falsePositiveRate The desired false positive probability.
     *                          Must be between 0.0 (exclusive) and 1.0 (exclusive).
     * 
     * @return The optimal size of the bit array (number of bits) for the Bloom filter.
     * 
     * @throws std::invalid_argument if maxElements is 0 or less.
     * @throws std::invalid_argument if falsePositiveRate is not in the range (0.0, 1.0).
     * 
     * @note The result is rounded up to the nearest integer to ensure the false positive
     *       rate does not exceed the specified value.
     */
    [[nodiscard]]
    static std::size_t calculateBitSize(std::size_t maxElements, double falsePositiveRate)
    {
        if (maxElements == 0) {
            throw std::invalid_argument("maxElements must be greater than 0, got " + std::to_string(maxElements));
        }
        if (falsePositiveRate <= 0.0 || falsePositiveRate >= 1.0) {
            throw std::invalid_argument("falsePositiveRate must be between 0 and 1, got " + std::to_string(falsePositiveRate));
        }
        double m = -((static_cast<double>(maxElements) * std::log(falsePositiveRate)) /
                     (std::pow(std::log(2.0), 2.0)));
        return static_cast<std::size_t>(std::ceil(m));
    }

    /**
     * @brief Calculates the optimal number of hash functions for the Bloom filter.
     * 
     * Uses the formula: k = (m/n) * ln(2), where:
     * - k is the number of hash functions
     * - m is the size of the bit array
     * - n is the expected number of elements
     * 
     * This formula minimizes the false positive probability for a given bit array size
     * and number of elements.
     * 
     * @param maxElements The maximum number of elements expected to be inserted (n)
     * @param bitArraySize The size of the bit array in bits (m)
     * @return The optimal number of hash functions, rounded up to the nearest integer
     */
    [[nodiscard]]
    static std::size_t calculateHashCount(std::size_t maxElements, std::size_t bitArraySize)
    {
        double k = (static_cast<double>(bitArraySize) / maxElements) * std::log(2.0);
        return static_cast<std::size_t>(std::ceil(k));
    }

private:
    template <typename F>
    /**
     * @brief Applies a function to each hash position for the given element.
     * 
     * This method computes multiple hash values for the provided element and applies
     * the given function to each resulting position in the bloom filter's bit array.
     * 
     * @tparam F The type of the callable object (function, lambda, functor)
     * @param element A span of bytes representing the element to hash
     * @param fn A callable that will be invoked for each computed hash position.
     *           The callable should accept a position index as its parameter.
     * @return true if all function invocations succeeded, false otherwise
     * 
     * @note The function processes the element through multiple hash functions to
     *       determine the positions in the bit array that correspond to this element.
     */
    bool forEachPosition(std::span<const std::byte> element, F&& fn) const
    {
        HashResult hashes = calculateHashes(element);
        for (std::size_t i = 0; i < this->hashCount; ++i) 
        {
            std::size_t position = (hashes.hash1 + (i + 1) * hashes.hash2) % this->bitArray.size();
            if (!fn(position)) 
            {
                return false;
            }
        }
        return true;
    }

    double falsePositiveRate;
    std::size_t maxElements;
    BitArray bitArray;
    std::size_t hashCount;
};

} // namespace bloom_filter

#endif // BLOOM_FILTER_HPP