use bloom_filter::BloomFilter;

fn main() {
    let mut bf = BloomFilter::new(1000, 0.01);
    println!("{bf}");

    let words = ["hello", "world", "bloom", "filter", "rust"];
    for word in &words {
        bf.add(word);
    }

    for word in &words {
        println!("  contains({word:?}) = {}", bf.contains(word));
    }
    println!("  contains(\"missing\") = {}", bf.contains(&"missing"));
}
