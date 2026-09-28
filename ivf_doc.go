package ivfq

// The sizing rule is the package's central claim. k = HoodK(N) =
// ceil(N/HoodTarget), so clusters hold about HoodTarget (1024) vectors
// whatever N is and cluster COUNT carries the growth. Under k = sqrt(N)
// both count and size grow as sqrt(N), and a query probing a fixed
// fraction of the clusters scores a number of candidates linear in N.
// Below hoodFloor x HoodTarget vectors a floor of min(hoodFloor,
// RoundSqrt(N)) clusters keeps a small index an index rather than one
// cluster.
//
// A default query probes DefaultNprobe(k) = min(max(98, k/32), max(16, k/4))
// clusters: a constant candidate budget, a k/32 recall floor above it, and
// a quarter-of-k ceiling under it. It is not constant in N and cannot be:
// flat IVF recall tracks the fraction of the corpus scanned, and no sizing
// changes that. What fixed-size clusters buy is a better index at whatever
// budget is chosen, and a far cheaper build.
//
// Fixed-size clusters are only viable with a sampled fit: at k = N/S,
// Lloyd's over all N is O(N^2/S) per pass. kmeans.Config.Fit trains on
// SampleSize(k) vectors (32 per centroid, floored at 4,096, measured within
// 1.03x of the full fit's MSE), seeds them with greedy k-means++, and
// assigns all N in one parallel pass, so a build costs O(SampleSize(k) x k
// x dims) to train plus O(N x k x dims) to assign. Every vector lands in
// exactly one cluster.
//
// After the build nothing in this package asks for a pass over N. Policy
// and MergeTarget are stated entirely on []ClusterStat, the per-cluster
// counts and radii an index already holds, so consulting them is O(k)
// arithmetic, and each answer is bounded by one cluster. SplitTarget names
// the cluster that has grown past SplitAbove (1.4x the target) or spread
// past twice the mean radius; MergeTarget names a cluster below MergeBelow
// (a quarter of the target) and the nearest neighbour with room to absorb
// it. Rebuilding an index is a caller's explicit decision.
