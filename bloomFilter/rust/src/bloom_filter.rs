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
        let bit_count = raw_bits + (BIT_ALIGNMENT - raw_bits % BIT_ALIGNMENT) % BIT_ALIGNMENT;
        let hash_count = optimal_hash_count(max_elements, bit_count);

        Self {
            bits: vec![0u64; bit_count / 64],
            bit_count,
            hash_count,
            max_elements,
            false_positive_rate,
        }
    }

    /// Inserts an item into the filter.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.hash_count {
            let idx = self.probe_index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// Returns `true` if the item *might* be in the set (possible false positive),
    /// or `false` if the item is *definitely* not in the set.
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (h1, h2) = self.hash_pair(item);
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

    /// Computes two independent 64-bit hashes using MurmurHash3-128 (first half)
    /// and xxHash64, mirroring the Python implementation's hash function choices.
    fn hash_pair<T: Hash>(&self, item: &T) -> (u64, u64) {
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

    /// Enhanced double hashing: position = (h1 + (i+1) * h2) mod bit_count.
    /// The (i+1) offset avoids the degenerate case where h2 == 0.
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
fn optimal_bit_count(n: usize, p: f64) -> usize {
    let m = -(n as f64 * p.ln()) / (LN_2 * LN_2);
    m.ceil() as usize
}

/// k = (m / n) * ln 2
fn optimal_hash_count(n: usize, m: usize) -> usize {
    let k = (m as f64 / n as f64) * LN_2;
    k.ceil() as usize
}

fn main() {
    let mut bf = BloomFilter::new(1000, 0.01);
    println!("{bf}");

    let words = ["hello", "world", "bloom", "filter", "rust"];
    for word in &words {
        bf.insert(word);
    }

    for word in &words {
        println!("  contains({word:?}) = {}", bf.contains(word));
    }
    println!("  contains(\"missing\") = {}", bf.contains(&"missing"));
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction ---

    #[test]
    fn new_filter_contains_nothing() {
        let bf = BloomFilter::new(100, 0.01);
        assert!(!bf.contains(&"anything"));
        assert!(!bf.contains(&""));
    }

    #[test]
    fn bit_count_is_64_aligned() {
        for n in [1, 7, 50, 1000, 9999] {
            let bf = BloomFilter::new(n, 0.01);
            assert_eq!(bf.bit_count() % 64, 0, "bit_count not 64-aligned for n={n}");
        }
    }

    #[test]
    fn lower_fpr_means_more_bits() {
        let coarse = BloomFilter::new(1000, 0.1);
        let fine = BloomFilter::new(1000, 0.001);
        assert!(
            fine.bit_count() > coarse.bit_count(),
            "lower FPR should require more bits"
        );
    }

    #[test]
    fn more_elements_means_more_bits() {
        let small = BloomFilter::new(100, 0.01);
        let large = BloomFilter::new(10_000, 0.01);
        assert!(large.bit_count() > small.bit_count());
    }

    #[test]
    #[should_panic(expected = "max_elements must be positive")]
    fn zero_elements_panics() {
        BloomFilter::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "false_positive_rate must be in (0.0, 1.0)")]
    fn fpr_zero_panics() {
        BloomFilter::new(100, 0.0);
    }

    #[test]
    #[should_panic(expected = "false_positive_rate must be in (0.0, 1.0)")]
    fn fpr_one_panics() {
        BloomFilter::new(100, 1.0);
    }

    #[test]
    #[should_panic(expected = "false_positive_rate must be in (0.0, 1.0)")]
    fn fpr_negative_panics() {
        BloomFilter::new(100, -0.5);
    }

    // --- Insert / Contains ---

    #[test]
    fn inserted_items_are_found() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let items = ["alpha", "bravo", "charlie", "delta", "echo"];
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "{item} should be found after insertion");
        }
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(&"duplicate");
        bf.insert(&"duplicate");
        bf.insert(&"duplicate");
        assert!(bf.contains(&"duplicate"));
    }

    #[test]
    fn empty_string_can_be_inserted() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(&"");
        assert!(bf.contains(&""));
    }

    #[test]
    fn single_element_filter() {
        let mut bf = BloomFilter::new(1, 0.01);
        assert!(!bf.contains(&"only"));
        bf.insert(&"only");
        assert!(bf.contains(&"only"));
    }

    #[test]
    fn works_with_integer_types() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(&42u64);
        bf.insert(&0i32);
        bf.insert(&-1i64);
        assert!(bf.contains(&42u64));
        assert!(bf.contains(&0i32));
        assert!(bf.contains(&-1i64));
        assert!(!bf.contains(&999u64));
    }

    #[test]
    fn works_with_byte_slices() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(&b"raw bytes".as_slice());
        assert!(bf.contains(&b"raw bytes".as_slice()));
        assert!(!bf.contains(&b"other bytes".as_slice()));
    }

    // --- False positive rate ---

    #[test]
    fn no_false_negatives() {
        let n = 5_000;
        let mut bf = BloomFilter::new(n, 0.01);
        for i in 0..n {
            bf.insert(&i);
        }
        for i in 0..n {
            assert!(bf.contains(&i), "false negative for {i}");
        }
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        let n = 10_000;
        let fpr = 0.05;
        let mut bf = BloomFilter::new(n, fpr);

        // Insert 0..n
        for i in 0..n {
            bf.insert(&i);
        }

        // Probe n..2n (none were inserted)
        let false_positives = (n..2 * n).filter(|i| bf.contains(i)).count();
        let observed_fpr = false_positives as f64 / n as f64;

        // Allow 2x the target rate as headroom for statistical variance
        assert!(
            observed_fpr < fpr * 2.0,
            "observed FPR {observed_fpr:.4} exceeds 2x target {fpr}"
        );
    }

    // --- Optimal parameter calculations ---

    #[test]
    fn optimal_bit_count_matches_formula() {
        // m = -(n * ln(p)) / (ln2)^2
        // For n=1000, p=0.01: m ≈ 9585.06 → ceil → 9586
        let m = optimal_bit_count(1000, 0.01);
        assert_eq!(m, 9586);
    }

    #[test]
    fn optimal_hash_count_matches_formula() {
        // k = (m/n) * ln2
        // For m=9600, n=1000: k ≈ 6.653 → ceil → 7
        let k = optimal_hash_count(1000, 9600);
        assert_eq!(k, 7);
    }

    // --- Display ---

    #[test]
    fn display_format() {
        let bf = BloomFilter::new(1000, 0.01);
        let s = format!("{bf}");
        assert!(s.contains("BloomFilter("));
        assert!(s.contains("max_elements=1000"));
        assert!(s.contains("fpr=0.010000"));
    }

    // --- Properties ---

    #[test]
    fn accessors_return_construction_values() {
        let bf = BloomFilter::new(500, 0.03);
        assert_eq!(bf.max_elements(), 500);
        assert!((bf.false_positive_rate() - 0.03).abs() < f64::EPSILON);
        assert!(bf.hash_count() > 0);
        assert!(bf.bit_count() > 0);
    }

    // --- Edge cases: near-extreme FPR ---

    #[test]
    fn very_low_fpr() {
        let bf = BloomFilter::new(100, 0.000001);
        assert!(bf.bit_count() > 0);
        assert!(bf.hash_count() > 0);
    }

    #[test]
    fn very_high_fpr() {
        let bf = BloomFilter::new(100, 0.999);
        // With FPR near 1.0 we still get a valid (tiny) filter
        assert!(bf.bit_count() > 0);
        assert!(bf.hash_count() >= 1);
    }

    // --- Distinct but similar strings ---

    #[test]
    fn distinguishes_similar_strings() {
        let mut bf = BloomFilter::new(1000, 0.001);
        bf.insert(&"abc");
        // These should almost certainly not collide at 0.1% FPR
        assert!(!bf.contains(&"abd"));
        assert!(!bf.contains(&"ab"));
        assert!(!bf.contains(&"abcd"));
    }

    // --- Large-scale insert ---

    #[test]
    fn handles_many_inserts() {
        let n = 50_000;
        let mut bf = BloomFilter::new(n, 0.01);
        for i in 0..n {
            bf.insert(&i);
        }
        // Spot-check a handful
        assert!(bf.contains(&0usize));
        assert!(bf.contains(&(n / 2)));
        assert!(bf.contains(&(n - 1)));
    }
}
