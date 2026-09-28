package ivfq

import (
	"cmp"
	"math"
	"slices"
)

const (
	bootstrapMin      = 64
	splitMin          = 8
	splitRadiusFactor = 2
	MergeMax          = 3

	// HoodTarget is the vectors-per-neighborhood the sizing rule aims at.
	// It is the ONE constant that makes a query's cost constant in N: three
	// things are proportional to it and they pull opposite ways — candidates
	// scored per query is nprobe x HoodTarget, pack bytes per hood is
	// HoodTarget x dims x 4 (512 KiB at 128 dims, one S3 GET's worth and the
	// unit the byte cache evicts), and the centroid set both the query and
	// the build must scan is N/HoodTarget.
	//
	// 1024 was CHOSEN, not swept: it makes the 1M index 977 hoods against the
	// 1,000 the old rule gave, so the 1M measurements are a controlled A/B on
	// one corpus rather than a comparison of two different indexes. A sweep
	// over {512, 2048, 4096} at 1M is the obvious follow-up and has not been
	// run. The reasoning: fixed-size neighborhoods.
	HoodTarget = 1024

	// CandidateBudget is how many candidates a default query is allowed to
	// score. 100,000 is what the sqrt-sized index spent at 1M for recall
	// 0.968, so the default is the old default, made constant in N.
	CandidateBudget = 100_000

	// budgetNprobe is the hood count CandidateBudget buys: a constant, and
	// the whole cost of a default query up to about a million vectors.
	budgetNprobe = (CandidateBudget + HoodTarget - 1) / HoodTarget

	// recallProbeDiv is the OTHER half of the default, and it exists because
	// a constant candidate budget does not hold recall and no sizing rule can
	// make it. Measured at 10M: the budget's 98 hoods return
	// recall@10 of 0.502, and reaching 0.93 takes 512. Flat IVF recall tracks
	// the FRACTION of the corpus scanned; fixed-size hoods improve the
	// constant in front of that — markedly — but do not change the law.
	//
	// So the default probes at least k/32. That makes a default query's cost
	// grow with N again, which is the price of a default that holds recall.
	// k/16 was the first value, set on the synthetic blob corpus before
	// the hoods carried 1-bit codes and an exact rerank. Measured on BIGANN
	// 10M with them:
	// k/16 = 634 probes recall@10 1.000 at p50 79 ms, k/32 = 317 probes
	// 0.997 at 49 ms, k/64 = 158 probes 0.990 at 31 ms. The default takes
	// the middle row; the recall controller can still raise it.
	recallProbeDiv = 32

	// probeCeilDiv and minProbeCeil cap the default at a quarter of the
	// hoods, never below 16. See DefaultNprobe.
	probeCeilDiv = 4
	minProbeCeil = 16

	// hoodFloor keeps a small namespace an index rather than a single pack:
	// below hoodFloor x HoodTarget vectors the rule would ask for one or two
	// hoods, which is a full scan with no frame for split/merge/reassign to
	// act on. Capped by RoundSqrt so a namespace at the bootstrap minimum
	// gets exactly the k it got before this rule existed.
	hoodFloor = 16

	// splitSizeNum/splitSizeDen is the size trigger: split a hood above 1.4x
	// the target. A hood that grows until it splits at T leaves two of T/2,
	// so under steady ingest sizes spread over [T/2, T] with density 1/x and
	// mean T/(2 ln 2): the trigger that makes the MEAN the target is
	// 2 ln 2 = 1.386x. At 2x (until 2026-09-27) the tier's 10M ingest ended
	// at 1.44x the target, 6,866 clusters where HoodK asks for 9,766.
	splitSizeNum = 7
	splitSizeDen = 5
	mergeSizeDiv = 4 // merge a hood below target / mergeSizeDiv
)

// DefaultNprobe is how many hoods a query probes when the caller does not
// name a number and no recall controller has tuned the namespace. It is
// also where the controller starts and what its ceiling is a multiple of
// (engine/recall_tune.go), so it is a starting point rather than the last
// word. Two bounds, and the smaller wins:
//
//	max(budgetNprobe, k/32)  the FLOOR: a constant candidate budget, raised
//	                         to a fixed fraction of the index once that is
//	                         bigger, because a constant budget does not hold
//	                         recall (see recallProbeDiv).
//	max(16, k/4)             the CEILING: never read more than a quarter of
//	                         the hoods by default once there are enough of
//	                         them to matter.
//
// The ceiling exists because the budget alone made the default an EXACT scan
// just below its own size: a 100,000-vector index has 98 hoods and the budget
// asks for 98, so the default read every one — 4.42 ms measured, for recall
// the index did not need. The quarter it now reads there is 24 hoods:
// recall@10 1.000 at 1.12 ms, 24,641 candidates against 100,000.
// Above ~400,000 vectors the ceiling is never the binding bound and 1M and 10M
// are untouched. Below 64 hoods it floors at 16, which is at or above k for
// every namespace the bootstrap threshold allows, so a small namespace still
// probes everything and answers exactly.
func DefaultNprobe(k int) int {
	return max(1, min(max(budgetNprobe, k/recallProbeDiv), max(minProbeCeil, k/probeCeilDiv)))
}

// HoodK is the sizing rule: k = ceil(total / HoodTarget), so hood size is
// constant in N and hood COUNT carries the growth. It replaces
// round(sqrt(total)), under which both count and size grew as sqrt(N) and
// candidates per query therefore grew linearly in N. Identical to the old rule for total <= 256; larger for
// everything above, by design.
func HoodK(total int) int {
	if total < 1 {
		return 1
	}
	k := (total + HoodTarget - 1) / HoodTarget
	if f := min(hoodFloor, RoundSqrt(total)); k < f {
		k = f
	}
	return max(1, min(k, total))
}

// HoodSize is the per-hood target HoodK implies at this total: HoodTarget
// once the namespace is big enough to want more than hoodFloor hoods, and
// total/k below that. Split and merge are stated against THIS, not against
// HoodTarget directly, so the small-namespace floor does not make every
// hood look tiny and merge the index into one pack.
func HoodSize(total int) int {
	return max(1, total/HoodK(total))
}

// SplitAbove is the band's MAXIMUM: the hood size past which the policy
// owes a split (SplitTarget's size trigger, 1.4x the target).
// It is stated here rather than inlined at the trigger because a hood no
// longer has to wait for a verb to learn it: a rewrite that leaves a hood
// above this splits it inside the rewrite, the way SPFresh's LIRE splits a
// posting inside the insert that overflowed it, and both readers of the
// rule have to agree on the same number. 1,432 at a million vectors.
func SplitAbove(total int) int {
	return HoodSize(total) * splitSizeNum / splitSizeDen
}

// MergeBelow is the hood size under which a hood is a wasted probe slot and
// should be absorbed by its nearest neighbor. Never below MergeMax, so the
// pre-existing "singleton hoods merge" behavior survives at any size.
func MergeBelow(total int) int {
	return max(MergeMax, HoodSize(total)/mergeSizeDiv)
}

// TotalCount sums a stats slice — the live vector count the split and merge
// thresholds are stated against. Both triggers are relative to the size the
// sizing rule wants at THIS total, so neither needs the policy to be told N.
func TotalCount(stats []ClusterStat) int {
	n := 0
	for _, s := range stats {
		n += s.Count
	}
	return n
}

// Policy implementations must be safe for concurrent use; the
// shipped policies are stateless.
//
// Every trigger past Bootstrap is stated on []ClusterStat and nothing else,
// and that is the point: a ClusterStat slice is read straight out of the
// manifest, so consulting the policy costs O(k) arithmetic and no pass over
// the vectors. There is deliberately NO whole-index trigger any more — the
// drift-gated full recluster this interface used to carry (ReclusterK, with
// its IndexStats of additive and live counts) was the last thing in the
// engine that could decide, on its own, to touch all N vectors. It is gone,
// and so is the regional re-fit that briefly replaced it; split, merge and
// reclaim are the whole vocabulary, and an explicit recluster is the only
// verb that still rebuilds a whole index when a human asks for it.
type Policy interface {
	// Bootstrap: no index exists; total = live vector count from the winner
	// scan. ok=false leaves the namespace unindexed this cycle. This is the
	// ONE pass over all N vectors the design allows.
	Bootstrap(total int) (k int, ok bool)
	// SplitTarget: consulted AFTER the append pass with post-append counts.
	SplitTarget(clusters []ClusterStat) (id int, ok bool)
}

// ClusterStat is one hood as the manifest records it. Count is the number of
// vectors the hood's PACK holds and Radius the mean distance from those
// members to its centroid, both written by whichever verb last rewrote the
// hood — so both are exact, delete-aware by construction (a deleted id is
// removed from its pack in the cycle that deletes it), and free to read.
// Radius == 0 means either "every member is the same point" or "emptied,
// awaiting reclaim"; either way it is not splittable evidence.
// Hits is query heat since the last cycle, and is 0 on a detached cycle.
type ClusterStat struct {
	ID, Count int
	Radius    float64
	Hits      int
}

// DefaultPolicy is the one production policy. The Policy interface exists
// so tests can substitute a stub that names a verb unconditionally; there
// is no second shipped policy. SplitTarget is answered by a verb bounded
// by a single hood:
//
//   - SplitTarget: size, then radius, for the hood one verb from the band.
type DefaultPolicy struct{}

func (DefaultPolicy) Bootstrap(total int) (int, bool) {
	if total < bootstrapMin {
		return 0, false
	}
	return HoodK(total), true
}

func (DefaultPolicy) SplitTarget(clusters []ClusterStat) (int, bool) {
	// Size trigger, and it wins: a hood past SplitAbove is
	// costing every query that probes it, whatever its radius and whether or
	// not it is hot. The fattest one goes first. This is the half of
	// fixed-size hoods that keeps them fixed — the radius rule below is the
	// shape trigger and is unchanged.
	//
	// Radius > 0 is the one exception and it is not a heuristic: a hood whose
	// members are all the same point CANNOT be split — 2-means over identical
	// vectors puts everything in one cluster and leaves the other empty, which
	// the reclaim verb then drops, which re-triggers the split. That is a
	// wasted structural verb every cycle, forever. Radius 0 means unsplittable,
	// so skip it however fat it is.
	if above := SplitAbove(TotalCount(clusters)); above > 0 {
		fat := -1
		for i, c := range clusters {
			if c.Count >= splitMin && c.Radius > 0 && c.Count > above &&
				(fat < 0 || c.Count > clusters[fat].Count) {
				fat = i
			}
		}
		if fat >= 0 {
			return clusters[fat].ID, true
		}
	}
	var total float64
	n, hits := 0, false
	for _, c := range clusters {
		if c.Count > 0 {
			total += c.Radius
			n++
		}
		hits = hits || c.Hits > 0
	}
	if n == 0 || total == 0 {
		return 0, false
	}
	mean := total / float64(n)
	for _, c := range clusters {
		if c.Count >= splitMin && c.Radius > splitRadiusFactor*mean && (!hits || c.Hits > 0) {
			return c.ID, true
		}
	}
	return 0, false
}

// RoundSqrt is the k heuristic: about √N neighborhoods of about √N vectors.
func RoundSqrt(total int) int {
	return int(math.Round(math.Sqrt(float64(total))))
}

// MergeTarget picks a tiny cluster and the nearest cluster to absorb it:
// the smallest tiny whose nearest neighbour stays at or under the split
// trigger with its rows. A merge past the trigger undoes the split that
// made the tiny, and the next cycle splits the host again: a lopsided
// 2-means split leaves a child under MergeBelow, the child's nearest
// cluster is its sibling, and the layout alternates split and merge on
// every cycle, forever. A tiny with no such neighbour stays until it grows.
// The size trigger is the one mirrored: a merge can still widen a host past
// the radius trigger. centroids[i] is stats[i]'s centroid; tiny and host
// index stats.
func MergeTarget(stats []ClusterStat, centroids [][]float32) (tiny, host int, ok bool) {
	if len(stats) < 2 {
		return 0, 0, false
	}
	// The merge threshold is relative to the size the sizing rule wants at
	// this total (MergeBelow), mirroring the split trigger from below.
	total := TotalCount(stats)
	below, above := MergeBelow(total), SplitAbove(total)
	var tinies []int
	for i, s := range stats {
		if s.Count >= 1 && s.Count <= below {
			tinies = append(tinies, i)
		}
	}
	slices.SortStableFunc(tinies, func(a, b int) int { return cmp.Compare(stats[a].Count, stats[b].Count) })
	smallest := smallestOther(stats)
	tries := 0
	for _, tiny := range tinies {
		if stats[tiny].Count+smallest(tiny) > above {
			continue // no cluster anywhere has room: skip the distance scan
		}
		// ponytail: at most mergeTries distance scans (k × dims each) per
		// call; a layout whose smallest tinies all sit beside full hosts
		// merges nothing this cycle. Index the centroids if that shows up.
		if tries++; tries > mergeTries {
			break
		}
		host := -1
		var best float32
		for i := range stats {
			if i == tiny {
				continue
			}
			d := L2Sq(centroids[tiny], centroids[i])
			if host < 0 || d < best {
				host, best = i, d
			}
		}
		if stats[host].Count+stats[tiny].Count <= above {
			return tiny, host, true
		}
	}
	return 0, 0, false
}

// mergeTries bounds the tinies one MergeTarget call measures distances for.
const mergeTries = 16

// smallestOther returns, for a cluster index, the smallest count among the
// other clusters: the most room any merge of it could find.
func smallestOther(stats []ClusterStat) func(i int) int {
	min1, min2 := -1, -1
	for i, s := range stats {
		switch {
		case min1 < 0 || s.Count < stats[min1].Count:
			min1, min2 = i, min1
		case min2 < 0 || s.Count < stats[min2].Count:
			min2 = i
		}
	}
	return func(i int) int {
		if i == min1 {
			return stats[min2].Count
		}
		return stats[min1].Count
	}
}

// MergeOwed reports an empty cluster (reclaim) or a tiny one with room
// somewhere to merge into. MergeTarget refuses a merge past the split
// trigger, so a tiny no cluster has room for owes nothing: counting it
// would buy a rebalancing cycle that declines every time.
func MergeOwed(stats []ClusterStat) bool {
	total := TotalCount(stats)
	below, above := MergeBelow(total), SplitAbove(total)
	smallest := smallestOther(stats)
	for i, s := range stats {
		if s.Count == 0 || s.Count <= below && s.Count+smallest(i) <= above {
			return true
		}
	}
	return false
}
