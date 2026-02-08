This is a C++ and Python implementation of a [Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)

It's a probabilistic data structure.  It can tell you that item *definetly* isn't in the data structure...or that it *could be* in the data structure.
The usefulness for this, is that the look is cheap and very memory efficient, so it can be used as an optimisation for expensive look ups.
For instance, a Bloom filter can be checked before doing an expensive check of file.  If it's not present, you save the lookup. If it might be present, you don't get the optimisation and have to perform the lookup, but the process still functions correctly.

It's basically two methods
1. `add` - a method that adds a item to the data structure.
2. `contains` - method that return true is it's possibly contained in the data structure, and false if it *definetly* isn't.

It only really requires three components with some logic.
The components are a container that can be used to access a bit array efficiently, and two independent hashing functions.
In this implementation, [xxHash](https://xxhash.com) and [MurmurHash](https://en.wikipedia.org/wiki/MurmurHash)

Whilst the math behind the Bloom fitler requires an [optimal number of hashing functions](https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions) in practice this isn't required.
Kirsch & Mitzenmacher found that you can use two independent hash functions using an approach called 'double' hash in their paper [Less Hashing, Same Performance](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)
