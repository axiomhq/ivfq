// The clustering half owns the pure geometry and decisions of the neighborhood
// index: k-means (plain, spherical, and sample-trained), splits,
// nearest-centroid lookup, and the rebalancing Policy with its shipped
// implementations. No I/O.
//
// The sizing rule is the package's central claim. k = HoodK(N) =
// ceil(N/HoodTarget), so hoods hold about HoodTarget (1024) vectors WHATEVER
// N is and hood COUNT carries the growth. The rule it replaced was
// k = round(sqrt(N)), under which count and size BOTH grew as sqrt(N): a
// query at the old default (10% of clusters) therefore scored a number of
// candidates linear in N — 100,000 at 1M and 1,002,530 at 10M, measured
//. The default is now DefaultNprobe(k) =
// min(max(98, k/32), max(16, k/4)): a constant candidate budget, a k/32
// recall FLOOR above it, and a quarter-of-k CEILING under it so the budget
// cannot become a full scan just below its own size.
// It is NOT constant in N and cannot be — a constant budget returns recall
// 0.502 at 10M, because flat IVF recall tracks the fraction of the corpus
// scanned and no hood sizing changes that. What hood sizing buys is a much
// better index at whatever budget is chosen, and a far cheaper build. Split
// (above 2x the target) and merge (below a quarter of it) hold hoods there
// between rebuilds; the radius rule is unchanged and is the other,
// shape-driven trigger. Below hoodFloor x HoodTarget vectors a floor keeps a
// small namespace an index rather than one pack, and it is capped by
// RoundSqrt — so for every total up to 256 the rule returns
// exactly the k it returned before fixed-size hoods existed.
//
// Fixed-size hoods are only viable WITH the sample bootstrap, which is why
// they landed as one change: at k = N/S, iterated Lloyd's over all N is
// O(N^2/S) per pass, worse than the sqrt-sized index it replaces. RunSampled
// fits the centroids on SampleSize(k) vectors (32 per centroid, floored at
// 4,096 — measured within 1.03x of the full fit's MSE), seeds them with
// GREEDY k-means++ (seedPlusPlus; plain D^2 sampling seeds outliers in high
// dimensions and made the hood distribution worse than a uniform init), and
// then assigns all N in ONE parallel pass. Build cost is therefore
// O(SampleSize(k) x k x dims) to train plus O(N x k x dims) to assign; the
// second term is still quadratic in N and is what a coarse centroid level
// would fix next.
//
// The assignment pass runs on GOMAXPROCS workers (assignAll) while
// everything around it stays serial, which is what keeps the result
// bit-identical whatever the machine width. It places each vector in exactly
// ONE hood: SPANN-style boundary replication was measured on 2026-09-05 and
// rejected, so the rule is gone.
//
// # No trigger in this package can ask for a pass over N (2026-09-05)
//
// Policy used to carry ReclusterK, which took an IndexStats of whole-index
// counts and could answer "rebuild all of it at k clusters". Compaction
// consulted it on every cycle and acted on it whenever the cluster count had
// drifted from HoodK by more than 2x. It is gone, and with it the last
// automatic O(N) pass in index maintenance: after the bootstrap, nothing
// short of an explicit recluster ever touches every vector
// again.
//
// What replaced it is stated entirely on []ClusterStat — per-hood counts and
// radii, which the manifest carries because every verb that rewrites a hood
// measures them over the members it just wrote. So the policy's whole input
// is O(k) arithmetic on data the compaction already holds, and the verbs that
// answer it are each bounded by ONE hood:
//
//   - SplitTarget handles the hood that has grown or spread past the band —
//     above SplitAbove (1.4x the target), or past splitRadiusFactor x the
//     mean radius. Since 2026-09-15 the split verb is no longer the only way
//     a hood is halved: SplitAbove is also applied inside the rewrite that
//     oversized a hood, so the verb handles what drifts out of band, not
//     what a merge or a patch just published there.
//   - The executor's merge absorbs a hood below target/mergeSizeDiv into its
//     nearest neighbor, whole, in one verb — which is also what brings k
//     back down on an index hollowed out by deletes, one hood per cycle.
//
// A REGIONAL re-fit (RegionSeed, and the executor's k-means over a seed hood
// and its nearest neighbors) sat between these two from 2026-09-05 until
// 2026-09-15. It was removed: it existed to buy back the cycles a hood far
// outside the band would otherwise spend halving itself, and the in-rewrite
// band split now keeps hoods from getting that far in the first place.
// Deletes are applied to the one hood that holds the id, so the per-hood
// counts a policy reads ARE the live counts, and merge is what a hollowed
// index converges on. Reassignment on patch used to be an opt-in policy
// (ReassignPolicy); since 2026-09-07 DefaultPolicy is the only shipped
// policy and it is always on.

package ivfq
