// Package ivfq is the clustering half of an IVF index in the FAISS sense
// ("IVF + quantizer"): the sizing rule that keeps clusters at about
// HoodTarget vectors, the centroid Tree a query probes, the split and merge
// rules that hold cluster sizes between rebuilds, the Policy that names the
// next cluster to split, and a binary codec for centroid sets and their
// deltas. It also holds the distance vocabulary every subpackage shares:
// Metric, Score, Dot, L2Sq and CosineSim.
//
// The other halves are subpackages: kmeans fits the centroids, rabitq
// gives every cluster a 1-bit candidate column with a scorer and an error
// bound, rank selects and fuses hits, recall tunes probe counts against a
// target, late scores multi-vector documents, sparse is the string-keyed
// sparse vector, codec stores rows at f16 and int8, and bench reads the
// standard ANN corpora. No I/O anywhere but bench.
package ivfq
