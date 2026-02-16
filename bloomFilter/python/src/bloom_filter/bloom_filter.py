from math import ceil, log

import mmh3
import xxhash
from bitarray import bitarray, frozenbitarray

BIT_ALIGNMENT = 64


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
        self._max_elements: int = max_elements
        self._false_positive_rate: float = false_positive_rate

        # Calculate the size of the bit array needed to achieve the desired false
        # positive rate for the given number of elements, and pad it to the nearest
        # multiple of BIT_ALIGNMENT for better performance.
        self._bit_array_size: int = self.calculate_bit_size(
            max_elements, false_positive_rate
        )
        self._bit_array_size += (
            BIT_ALIGNMENT - (self._bit_array_size % BIT_ALIGNMENT)
        ) % BIT_ALIGNMENT
        self._bit_array: bitarray = bitarray(self._bit_array_size)
        self._bit_array.setall(0)

        # Calculate the number of hash functions needed to achieve the desired false
        # positive rate for the given number of elements.
        self._hash_count: int = self.calculate_hash_count(
            self._max_elements, self._bit_array_size
        )

    def add(self, item: str) -> None:
        """Add an item to the Bloom filter.

        This method adds an item to the Bloom filter by setting multiple bits in the
        bit array. The number of bits set is determined by the hash_count parameter.
        Each bit position is calculated using a combination of hash functions.

        Args:
            item (str): The item to add to the Bloom filter.

        Returns:
            None
        """
        hash1, hash2 = self._calculate_hashes(item)

        # Add 1 to i to handle the degenerate case where hash2 is 0
        def position(i: int) -> int:
            return (hash1 + (i + 1) * hash2) % self._bit_array_size

        for i in range(self._hash_count):
            self._bit_array[position(i)] = 1

    def contains(self, item: str) -> bool:
        """
        Check if an item might be in the Bloom filter.

        Args:
            item (str): The item to check for membership.

        Returns:
            bool: False if the item is definitely not in the set.
                  True if the item might be in the set (possibility of false positives).

        Note:
            A return value of False guarantees the item was never added.
            A return value of True means the item was probably added, but could be a false positive.
        """
        hash1, hash2 = self._calculate_hashes(item)

        # Add 1 to i to handle the degenerate case where hash2 is 0
        def position(i: int) -> int:
            return (hash1 + (i + 1) * hash2) % self._bit_array_size

        return all(self._bit_array[position(i)] for i in range(self._hash_count))

    def __contains__(self, item: str) -> bool:
        """Enable the use of the 'in' operator for membership testing.

        This method allows you to use the syntax `item in bloom_filter` to check if
        an item is possibly in the Bloom filter. It internally calls the `contains`
        method to perform the membership test.

        Args:
            item (str): The item to check for membership.

        Returns:
            bool: False if the item is definitely not in the set.
                  True if the item might be in the set (possibility of false positives).
        """
        return self.contains(item)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the BloomFilter instance.

        Returns:
            str: A string containing the BloomFilter's configuration parameters including
                max_elements, false_positive_rate, bit_array_size, hash_count, and the
                current state of the bit array.
        """
        return (
            f"BloomFilter(max_elements={self._max_elements}, "
            f"false_positive_rate={self._false_positive_rate:.6f}, "
            f"bit_array_size={self._bit_array_size}, "
            f"hash_count={self._hash_count}, "
            f"bitarray={self._bit_array})"
        )

    def _calculate_hashes(self, item: str) -> tuple[int, int]:
        """
        Calculate two independent hash values for the given item.

        This method uses two different hashing algorithms (MurmurHash3 and xxHash)
        to generate two independent hash values that will be used for double hashing
        in the Bloom filter.

        Args:
            item (str): The string item to be hashed.

        Returns:
            tuple[int, int]: A tuple containing two hash values:
                - hash1: The first 64-bit hash value from MurmurHash3
                - hash2: The 64-bit hash value from xxHash

        Note:
            These hash values are used in conjunction with double hashing to
            generate multiple hash functions for the Bloom filter, reducing
            the need for multiple independent hash functions.
        """
        # Pylint disable is necessary here because the mmh3.hash64 function returns a
        # tuple, but the type hint in the mmh3 stub file does not reflect this correctly.
        # pylint: disable=unsubscriptable-object
        hash1: int = mmh3.hash64(item, signed=False)[0]
        hash2: int = xxhash.xxh64(item).intdigest()
        return (hash1, hash2)

    @property
    def max_elements(self) -> int:
        """
        Get the maximum number of elements the Bloom filter was designed for.

        Returns:
            int: The maximum number of elements that can be added to the filter
                 while maintaining the desired false positive rate.
        """
        return self._max_elements

    @property
    def false_positive_rate(self) -> float:
        """
        Get the theoretical false positive probability for the Bloom filter.

        Returns:
            float: The false positive probability based on the current number of items,
                   bit array size, and number of hash functions.
        """
        return self._false_positive_rate

    @property
    def bit_array_size(self) -> int:
        """
        Get the size of the bit array.

        Returns:
            int: The size of the bit array in bits.
        """
        return self._bit_array_size

    @property
    def hash_count(self) -> int:
        """Get the number of hash functions used by the Bloom filter.

        Returns:
            int: The number of hash functions configured for this Bloom filter instance.
        """
        return self._hash_count

    @property
    def bit_array(self) -> frozenbitarray:
        """
        Get a frozen copy of the internal bit array.

        Returns:
            frozenbitarray: An immutable copy of the bloom filter's bit array.
                This prevents external modification of the filter's internal state.
        """
        return frozenbitarray(self._bit_array)

    @staticmethod
    def calculate_bit_size(max_elements: int, false_positive_rate: float) -> int:
        """
        Calculate the optimal bit array size for a Bloom filter.

        Args:
            max_elements (int): The maximum number of elements expected to be stored in the filter.
            false_positive_rate (float): The desired false positive probability (between 0 and 1).

        Returns:
            int: The optimal size of the bit array to achieve the desired false positive rate.

        Formula:
            m = -(n * ln(p)) / (ln(2))^2
            where:
                m = bit array size
                n = maximum number of elements
                p = false positive rate
        """
        # The use of log(2) rather than a precomputed constant is for readability.
        # This static method is only called once during initialization, so the
        # performance impact is negligible.
        m: float = -(max_elements * log(false_positive_rate) / log(2) ** 2)
        return int(ceil(m))

    @staticmethod
    def calculate_hash_count(max_elements: int, bit_array_size: int) -> int:
        """
        Calculate the optimal number of hash functions for a Bloom filter.

        This function determines the ideal number of hash functions (k) based on the
        size of the bit array (m) and the expected number of elements (n) using the
        formula: k = (m/n) * ln(2)

        Args:
            max_elements (int): The maximum number of elements expected to be inserted
                               into the Bloom filter.
            bit_array_size (int): The size of the bit array in bits.

        Returns:
            int: The optimal number of hash functions to use, rounded up to the
                 nearest integer.

        Note:
            This calculation minimizes the false positive probability for the given
            bit array size and expected number of elements.
        """
        # The use of log(2) rather than a precomputed constant is for readability.
        # This static method is only called once during initialization, so the
        # performance impact is negligible.
        k: float = (bit_array_size / max_elements) * log(2)
        return int(ceil(k))
