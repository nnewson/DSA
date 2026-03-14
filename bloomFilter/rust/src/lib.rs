use std::fmt;
use std::hash::{Hash, Hasher};

const BIT_ALIGNMENT: usize = 64;
const LN_2: f64 = std::f64::consts::LN_2;

/// A space-efficient probabilistic data structure for set membership testing.
///
/// A Bloom filter can tell you with certainty that an element is *not* in the set,
/// but membership queries may produce false positives. It never produces false negatives.
///
/// Uses enhanced double hashing (h1 + (i+1)*h2) with MurmurHash3 and xxHash64
/// to simulate `k` independent hash functions from only two base hashes.
pub struct BloomFilter {
    bits: Vec<u64>,
    bit_count: usize,
    hash_count: usize,
    max_elements: usize,
    false_positive_rate: f64,
}

impl BloomFilter {
    /// Creates a new Bloom filter sized for `max_elements` at the given `false_positive_rate`.
    ///
    /// # Panics
    /// Panics if `max_elements` is 0 or `false_positive_rate` is not in (0.0, 1.0).
    pub fn new(max_elements: usize, false_positive_rate: f64) -> Self {
        assert!(max_elements > 0, "max_elements must be positive");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "false_positive_rate must be in (0.0, 1.0), got {false_positive_rate}"
        );

        let raw_bits = optimal_bit_count(max_elements, false_positive_rate);
        // Pad to next multiple of BIT_ALIGNMENT for word-aligned storage
        let bit_count = raw_bits.next_multiple_of(BIT_ALIGNMENT);
        let hash_count = optimal_hash_count(max_elements, bit_count);

        Self {
            bits: vec![0u64; bit_count / 64],
            bit_count,
            hash_count,
            max_elements,
            false_positive_rate,
        }
    }

    /// Adds an item to the filter.
    pub fn add<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = hash_pair(item);
        for i in 0..self.hash_count {
            let idx = self.probe_index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// Returns `true` if the item *might* be in the set (possible false positive),
    /// or `false` if the item is *definitely* not in the set.
    #[must_use]
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        (0..self.hash_count).all(|i| {
            let idx = self.probe_index(h1, h2, i);
            self.get_bit(idx)
        })
    }

    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    pub fn false_positive_rate(&self) -> f64 {
        self.false_positive_rate
    }

    /// Enhanced double hashing: position = (h1 + (i+1) * h2) mod bit_count.
    /// The (i+1) offset avoids the degenerate case where i == 0.
    #[inline]
    fn probe_index(&self, h1: u64, h2: u64, i: usize) -> usize {
        let combined = h1.wrapping_add((i as u64 + 1).wrapping_mul(h2));
        (combined % self.bit_count as u64) as usize
    }

    #[inline]
    fn set_bit(&mut self, index: usize) {
        self.bits[index / 64] |= 1 << (index % 64);
    }

    #[inline]
    fn get_bit(&self, index: usize) -> bool {
        self.bits[index / 64] & (1 << (index % 64)) != 0
    }
}

impl fmt::Display for BloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BloomFilter(max_elements={}, fpr={:.6}, bits={}, hashes={})",
            self.max_elements, self.false_positive_rate, self.bit_count, self.hash_count
        )
    }
}

/// m = -(n * ln(p)) / (ln 2)^2
pub fn optimal_bit_count(n: usize, p: f64) -> usize {
    let m = -(n as f64 * p.ln()) / (LN_2 * LN_2);
    m.ceil() as usize
}

/// Computes two independent 64-bit hashes using MurmurHash3-128 (first half)
/// and xxHash64, mirroring the Python implementation's hash function choices.
fn hash_pair<T: Hash>(item: &T) -> (u64, u64) {
    let h1 = {
        let mut hasher = mur3::Hasher128::with_seed(0);
        item.hash(&mut hasher);
        hasher.finish128().0
    };
    let h2 = {
        let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);
        item.hash(&mut hasher);
        hasher.finish()
    };
    (h1, h2)
}

/// k = (m / n) * ln 2
pub fn optimal_hash_count(n: usize, m: usize) -> usize {
    let k = (m as f64 / n as f64) * LN_2;
    k.ceil() as usize
}
