// Package ivfq is the clustering half of an IVF index in the FAISS sense
// ("IVF + quantizer"): the sizing rule that keeps clusters at about
// HoodTarget vectors, the centroid trees a query probes, the split and merge
// rules that hold cluster sizes between rebuilds, the Policy that names the
// next cluster to split, and a binary codec for centroid sets and their
// deltas. It also holds the distance vocabulary every subpackage shares:
// Metric, Score, Dot, L2Sq and CosineSim.
//
// The sizing rule is the package's central claim: k = HoodK(N) =
// ceil(N/HoodTarget), so clusters hold about HoodTarget vectors whatever N
// is and cluster COUNT carries the growth. A default query probes
// DefaultNprobe(k) clusters, which is not constant in N and cannot be: flat
// IVF recall tracks the fraction of the corpus scanned, and no sizing
// changes that. What fixed-size clusters buy is a better index at whatever
// budget is chosen, and a far cheaper build, because kmeans fits them on a
// sample. After the build nothing here asks for a pass over N: Policy and
// MergeTargets read only per-cluster counts, radii and centroids, which an
// index already holds, and rebuilding an index is a caller's explicit
// decision.
//
// The other halves are subpackages: kmeans fits the centroids, rabitq
// gives every cluster a 1-bit candidate column with a scorer and an error
// bound, rank selects and fuses hits by int id, codec stores rows at f16
// and int8, vector reads flat float32, f16, int8 or uint8 input as float32
// rows, and bench reads the standard ANN corpora. No I/O anywhere but
// bench.
package ivfq
