package ivfq

import "testing"

// TestMergeTargetKeepsTheHostUnderTheSplitTrigger pins the merge's
// hysteresis: a tiny cluster whose nearest neighbour would pass the split
// trigger with its rows is not merged (the next cycle would split the host
// again, and the layout would alternate split and merge forever); the next
// smallest tiny with room goes instead, and a tiny with room merges into
// its nearest neighbour.
func TestMergeTargetKeepsTheHostUnderTheSplitTrigger(t *testing.T) {
	// Slots: 0 a tiny beside a full sibling (1), 2 a tiny beside a
	// roomy cluster (3), and filler far away.
	counts := []int{10, 80, 12, 40}
	centroids := [][]float32{{0, 0}, {1, 0}, {100, 0}, {101, 0}}
	for i := range 11 {
		counts = append(counts, 78)
		centroids = append(centroids, []float32{float32(1000 + 100*i), 1000})
	}
	stats := func(counts []int) []ClusterStat {
		s := make([]ClusterStat, len(counts))
		for i, n := range counts {
			s[i] = ClusterStat{ID: i, Count: n, Radius: 1}
		}
		return s
	}
	st := stats(counts)
	total := TotalCount(st)
	below, above := MergeBelow(total), SplitAbove(total)
	if counts[0] > below || counts[2] > below || counts[0]+counts[1] <= above || counts[1] > above || counts[2]+counts[3] > above {
		t.Fatalf("setup: total %d, merge below %d, split above %d do not fit the counts %v", total, below, above, counts[:4])
	}

	if tiny, host, ok := MergeTarget(st, centroids); !ok || tiny != 2 || host != 3 {
		t.Fatalf("merge %d into %d (ok %v); want the second tiny, 2, into its roomy neighbour 3", tiny, host, ok)
	}
	// Only the blocked tiny left: no merge.
	blocked := append([]int{}, counts...)
	blocked[2] = 30
	if tiny, host, ok := MergeTarget(stats(blocked), centroids); ok {
		t.Fatalf("merged %d into %d: the host would pass the split trigger (%d + %d > %d)", tiny, host, blocked[tiny], blocked[host], above)
	}
	if !MergeOwed(stats(blocked)) {
		// Room exists (the filler), only not beside the tiny: owed, and
		// the rebalancing cycle declines once.
		t.Fatal("a tiny with room somewhere is not owed")
	}
	// No room anywhere (fewer clusters than the sizing rule wants, so the
	// hosts sit past the trigger): nothing owed.
	full := []int{10, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80}
	if st := stats(full); MergeOwed(st) {
		t.Fatalf("owed a merge no cluster has room for (above %d)", SplitAbove(TotalCount(st)))
	}
	if _, _, ok := MergeTarget(stats(full), centroids[:len(full)]); ok {
		t.Fatal("merged a tiny no cluster has room for")
	}
	// Room beside the smallest tiny: it merges into its nearest neighbour.
	roomy := append([]int{}, counts...)
	roomy[1] = 60
	if tiny, host, ok := MergeTarget(stats(roomy), centroids); !ok || tiny != 0 || host != 1 {
		t.Fatalf("merge %d into %d (ok %v); want the smallest tiny, 0, into its nearest neighbour 1", tiny, host, ok)
	}
	// An empty cluster is owed a reclaim.
	empty := append([]int{}, full...)
	empty[0] = 0
	if !MergeOwed(stats(empty)) {
		t.Fatal("an empty cluster is not owed")
	}
}
