#include <cmath>
#include <cstddef>
#include <vector>

#include "xxhash.h"
#include "MurmurHash3.h"

constexpr size_t BIT_ALIGNMENT = 64;

class bloomFilter {
public:
    bloomFilter(size_t maxElements, double falsePositiveRate)
    {
        if (maxElements == 0) {
            throw std::invalid_argument("maxElements must be greater than 0, got " + std::to_string(maxElements));
        }
        if (falsePositiveRate <= 0.0 || falsePositiveRate >= 1.0) {
            throw std::invalid_argument("falsePositiveRate must be between 0 and 1, got " + std::to_string(falsePositiveRate));
        }

        // Store the maximum number of elements and the desired false positive rate
        // for reference.    
        this->maxElements = maxElements;
        this->falsePositiveRate = falsePositiveRate;
        
        // Calculate the size of the bit array needed to achieve the desired false
        // positive rate for the given number of elements, and pad it to the nearest
        // multiple of BIT_ALIGNMENT for better performance.
        this->bitArraySize = this->calculateBitSize(
            maxElements, falsePositiveRate
        );
        this->bitArraySize += (
            BIT_ALIGNMENT - (this->bitArraySize % BIT_ALIGNMENT)
        ) % BIT_ALIGNMENT;
        this->bitArray = std::vector<bool>(this->bitArraySize, false);

        // Calculate the number of hash functions needed to achieve the desired false
        // positive rate for the given number of elements.
        this->hashCount = this->calculateHashCount(
            this->maxElements, this->bitArraySize
        );
    }

    void add(const std::byte* element, size_t length)
    {
        for (size_t i = 0; i < this->hashCount; ++i) {
            size_t combinedHash = this->calculateCombinedHash(element, length, i + 1);
            this->bitArray[combinedHash] = true;
        }
    }

    bool contains(const std::byte* element, size_t length) const
    {
        for (size_t i = 0; i < this->hashCount; ++i) {
            size_t combinedHash = this->calculateCombinedHash(element, length, i + 1);
            if (!this->bitArray[combinedHash]) {
                return false;
            }
        }
        return true;
    }

    size_t getMaxElements() const 
    {
        return maxElements; 
    }

    double getFalsePositiveRate() const 
    { 
        return falsePositiveRate; 
    }

    size_t getBitArraySize() const 
    { 
        return bitArraySize; 
    }

    size_t getHashCount() const 
    { 
        return hashCount; 
    }

    const std::vector<bool>& getBitArray() const 
    { 
        return bitArray; 
    }

    size_t calculateCombinedHash(const std::byte* element, size_t length, size_t i) const
    {
        uint64_t hash1 = XXH64(element, length, i);

        uint32_t mmh3Hash[4];
        MurmurHash3_x64_128(element, length, i, mmh3Hash);
        uint64_t hash2 = (static_cast<uint64_t>(mmh3Hash[0]) << 32) | mmh3Hash[1];

        return (hash1 + i * hash2) % this->bitArraySize;
    }

    static size_t calculateBitSize(size_t maxElements, double falsePositiveRate)
    {
        double m = -((static_cast<double>(maxElements) * std::log(falsePositiveRate)) / 
                     (std::pow(std::log(2.0), 2.0)));
        return static_cast<size_t>(std::ceil(m) );
    }

    static size_t calculateHashCount(size_t maxElements, size_t bitArraySize)
    {
        double k = (static_cast<double>(bitArraySize) / maxElements) * std::log(2.0);
        return static_cast<size_t>(std::ceil(k));
    }

private:
    double falsePositiveRate;
    size_t maxElements;
    size_t bitArraySize;
    size_t hashCount;

    std::vector<bool> bitArray;
};
