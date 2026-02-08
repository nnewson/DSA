#include <cstddef>
#include <vector>

#include "xxhash.h"
#include "MurmurHash3.h"

class bloomFilter {
public:
    bloomFilter(size_t maxElements, double falsePositiveRate)
    {

    }

private:
    std::vector<bool> bitArray;
};
