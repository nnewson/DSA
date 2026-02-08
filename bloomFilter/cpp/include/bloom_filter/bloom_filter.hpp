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

class BloomFilter {
public:
    BloomFilter(std::size_t maxElements, double falsePositiveRate)
        : falsePositiveRate(falsePositiveRate),
          maxElements(maxElements),
          bitArray(calculateBitSize(maxElements, falsePositiveRate)),
          hashCount(calculateHashCount(maxElements, bitArray.size()))    
    {
    }

    void add(std::span<const std::byte> element)
    {
        for (std::size_t i = 0; i < this->hashCount; ++i) {
            std::size_t combinedHash = this->calculateCombinedHash(element, i + 1);
            this->bitArray.set(combinedHash);
        }
    }

    [[nodiscard]]
    bool contains(std::span<const std::byte> element) const
    {
        for (std::size_t i = 0; i < this->hashCount; ++i) {
            std::size_t combinedHash = this->calculateCombinedHash(element, i + 1);
            if (!this->bitArray.test(combinedHash)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    std::size_t getMaxElements() const noexcept
    {
        return maxElements; 
    }

    [[nodiscard]]
    double getFalsePositiveRate() const noexcept
    { 
        return falsePositiveRate; 
    }

    [[nodiscard]]
    std::size_t getBitArraySize() const noexcept
    { 
        return bitArray.size(); 
    }

    [[nodiscard]]
    std::size_t getHashCount() const noexcept
    { 
        return hashCount; 
    }

    [[nodiscard]]
    const BitArray& getBitArray() const noexcept
    { 
        return bitArray; 
    }

    [[nodiscard]]
    std::size_t calculateCombinedHash(std::span<const std::byte> element, std::size_t i) const
    {
        uint64_t hash1 = XXH64(element.data(), element.size(), i);

        uint32_t mmh3Hash[4];
        MurmurHash3_x64_128(element.data(), element.size(), i, mmh3Hash);
        uint64_t hash2 = (static_cast<uint64_t>(mmh3Hash[0]) << 32) | mmh3Hash[1];

        return (hash1 + i * hash2) % this->bitArray.size();
    }

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

    [[nodiscard]]
    static std::size_t calculateHashCount(std::size_t maxElements, std::size_t bitArraySize)
    {
        double k = (static_cast<double>(bitArraySize) / maxElements) * std::log(2.0);
        return static_cast<std::size_t>(std::ceil(k));
    }

private:
    double falsePositiveRate;
    std::size_t maxElements;
    BitArray bitArray;
    std::size_t hashCount;
};

} // namespace bloom_filter

#endif // BLOOM_FILTER_HPP