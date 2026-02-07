from math import log, ceil
from bitarray import bitarray

import mmh3
import xxhash

BYTE_ALIGNMENT = 8


class BloomFilter:
    def __init__(self, max_elements: int, false_positive_rate: float):
        if max_elements <= 0:
            raise ValueError(f"max_elements must be positive, got {max_elements}")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError(
                f"false_positive_rate must be in (0.0, 1.0), got {false_positive_rate}"
            )

        # Store the maximum number of elements and the desired false positive rate
        # for reference.
        self._max_elements = max_elements
        self._false_positive_rate = false_positive_rate

        # Calculate the size of the bit array needed to achieve the desired false
        # positive rate for the given number of elements, and pad it to the nearest
        # multiple of BYTE_ALIGNMENT for better performance.
        self._bit_array_size = self._calculate_bit_size(
            max_elements, false_positive_rate
        )
        bit_alignment = BYTE_ALIGNMENT * 8  # Convert bytes to bits
        self._bit_array_size += bit_alignment - (self._bit_array_size % bit_alignment)
        self._bit_array = bitarray(self._bit_array_size)
        self._bit_array.setall(0)

        # Calculate the number of hash functions needed to achieve the desired false
        # positive rate for the given number of elements.
        self._hash_count = self._calculate_hash_count(
            self._max_elements, self._bit_array_size
        )

    def add(self, item: str) -> None:
        for i in range(self._hash_count):
            combined_hash = self._calculate_combined_hash(item, i)
            self._bit_array[combined_hash] = 1

    def contains(self, item: str) -> bool:
        for i in range(self._hash_count):
            combined_hash = self._calculate_combined_hash(item, i)
            if not self._bit_array[combined_hash]:
                return False
        return True

    def _calculate_combined_hash(self, item: str, i: int) -> int:
        # Use a combination of two different hash functions to generate multiple
        # hash values for the item. This helps to reduce the chances of hash
        # collisions and improve the distribution of bits in the bit array.

        # Additionally, adding the index `i` to the seed of the second hash function
        # ensures that we get different hash values for each iteration, further
        # improving the distribution.

        # FYI: Pylint seems have an issue with the below being "unscriptable".
        # However, it always reutrn a tuple when signed is False, so we can safely
        # disable the warning here.
        # pylint: disable=unsubscriptable-object
        hash1 = mmh3.hash64(item, i, signed=False)[0] % self._bit_array_size
        hash2 = xxhash.xxh64(item, seed=i).intdigest() % self._bit_array_size
        return (hash1 + i * hash2) % self._bit_array_size

    @staticmethod
    def _calculate_bit_size(max_elements: int, false_positive_rate: float) -> int:
        m = -(max_elements * log(false_positive_rate) / log(2) ** 2)
        return int(ceil(m))

    @staticmethod
    def _calculate_hash_count(max_elements: int, bit_array_size: int) -> int:
        k = (bit_array_size / max_elements) * log(2)
        return int(ceil(k))
