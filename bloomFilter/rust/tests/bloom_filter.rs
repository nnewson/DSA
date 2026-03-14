use bloom_filter::{BloomFilter, optimal_bit_count, optimal_hash_count};

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
        bf.add(item);
    }
    for item in &items {
        assert!(bf.contains(item), "{item} should be found after insertion");
    }
}

#[test]
fn duplicate_insert_is_idempotent() {
    let mut bf = BloomFilter::new(100, 0.01);
    bf.add(&"duplicate");
    bf.add(&"duplicate");
    bf.add(&"duplicate");
    assert!(bf.contains(&"duplicate"));
}

#[test]
fn empty_string_can_be_inserted() {
    let mut bf = BloomFilter::new(100, 0.01);
    bf.add(&"");
    assert!(bf.contains(&""));
}

#[test]
fn single_element_filter() {
    let mut bf = BloomFilter::new(1, 0.01);
    assert!(!bf.contains(&"only"));
    bf.add(&"only");
    assert!(bf.contains(&"only"));
}

#[test]
fn works_with_integer_types() {
    let mut bf = BloomFilter::new(100, 0.01);
    bf.add(&42u64);
    bf.add(&0i32);
    bf.add(&-1i64);
    assert!(bf.contains(&42u64));
    assert!(bf.contains(&0i32));
    assert!(bf.contains(&-1i64));
    assert!(!bf.contains(&999u64));
}

#[test]
fn works_with_byte_slices() {
    let mut bf = BloomFilter::new(100, 0.01);
    bf.add(&b"raw bytes".as_slice());
    assert!(bf.contains(&b"raw bytes".as_slice()));
    assert!(!bf.contains(&b"other bytes".as_slice()));
}

// --- False positive rate ---

#[test]
fn no_false_negatives() {
    let n = 5_000;
    let mut bf = BloomFilter::new(n, 0.01);
    for i in 0..n {
        bf.add(&i);
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
        bf.add(&i);
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
    bf.add(&"abc");
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
        bf.add(&i);
    }
    // Spot-check a handful
    assert!(bf.contains(&0usize));
    assert!(bf.contains(&(n / 2)));
    assert!(bf.contains(&(n - 1)));
}
