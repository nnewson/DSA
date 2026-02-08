#include <cassert>
#include <iostream>

#include "bloom_filter/bloom_filter.hpp"

int main() {
    bloomFilter bf(1000, 0.01);
    std::cout << "bloomFilter created successfully" << std::endl;
    return 0;
}
