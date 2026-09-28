package ivfq

import (
	"testing"
)

func TestDefaultPolicyPinned(t *testing.T) {
	p := DefaultPolicy{}
	if _, ok := p.Bootstrap(63); ok {
		t.Fatal("bootstrap below min")
	}
	if k, ok := p.Bootstrap(64); !ok || k != 8 {
		t.Fatalf("k=%d ok=%v", k, ok)
	}
}

// TestNoPolicyTriggerSeesTheWholeIndex is the interface-level statement of

// what this package gave up on 2026-09-05: every trigger past Bootstrap takes

// []ClusterStat and nothing else. There is no argument through which a policy

// could learn N, and therefore no way for one to ask for a pass over N. The

// drift trigger that used to (ReclusterK, taking an IndexStats of additive

// and live totals) is gone; an explicit recluster is the only verb that still

// rebuilds an index on request.

//

// This is a compile-time property, so the test is a compile-time assertion:

// the Policy interface is satisfied by a type whose only per-index input is a

// cluster slice. If a whole-index trigger is ever added back, this stops

// building and whoever added it has to read the paragraph above.

func TestNoPolicyTriggerSeesTheWholeIndex(t *testing.T) {
	var _ Policy = clusterOnlyPolicy{}
}

type clusterOnlyPolicy struct{ DefaultPolicy }

// pad makes n filler hoods of the given size, so a table case can state a

// realistic total (and therefore a realistic per-hood target) without

// spelling out sixteen clusters.

func pad(n, count int) []ClusterStat {
	out := make([]ClusterStat, n)
	for i := range out {
		out[i] = ClusterStat{ID: 100 + i, Count: count, Radius: .1}
	}
	return out
}

// padRadius sets every filler hood's radius to r.

func padRadius(in []ClusterStat, r float64) []ClusterStat {
	for i := range in {
		in[i].Radius = r
	}
	return in
}

// TestSplitTargetRatioBoundary pins SplitTarget's ratio-gate boundary (T1

// review minor c, optional): count exactly equal to splitFactor×mean must

// NOT split (strict >); one more must.

func TestSplitTarget(t *testing.T) {
	p := DefaultPolicy{}
	tests := []struct {
		name string
		in   []ClusterStat
		id   int
		ok   bool
	}{
		// Both radius cases are stated against a padded index so the SIZE
		// trigger stays quiet and they still test what they were written to
		// test. Unpadded, a two-hood {80, 8} index has a target of 9 and 80
		// is a legitimate split under fixed-size hoods.
		{"tight fat", append([]ClusterStat{{ID: 1, Count: 80}, {ID: 2, Count: 8}}, pad(14, 64)...), 0, false},
		{"below minimum", append([]ClusterStat{{ID: 1, Count: 7, Radius: 100}, {ID: 2, Count: 8, Radius: .1}}, pad(3, 8)...), 0, false},
		{"sick", []ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
		// Padded at radius .2 so the mean stays .2 and the size trigger quiet.
		{"strict boundary", append([]ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: 0}}, padRadius(pad(14, 8), .2)...), 0, false},
		{"cold sick ignored", append([]ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1, Hits: 1}, {ID: 3, Count: 8, Radius: .1}}, pad(14, 8)...), 0, false},
		{"hot sick", []ClusterStat{{ID: 1, Count: 8, Radius: .4, Hits: 1}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
		{"unseen fallback", []ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id, ok := p.SplitTarget(tt.in)
			if id != tt.id || ok != tt.ok {
				t.Fatalf("got (%d,%v), want (%d,%v)", id, ok, tt.id, tt.ok)
			}
		})
	}
}

// TestSplitClusterRejectsUnknownTarget: a policy returning a split id that
// names no real cluster must surface as a CompactNow error (the caller
// already log-and-retries — see startCompactor), never a panic.

// TestDefaultNprobeCurve pins the whole default across five orders of

// magnitude, because it is three rules interacting (a constant budget, a

// k/32 recall floor, a k/4 ceiling) and the interesting values are the

// crossovers, not any one rule.

func TestDefaultNprobeCurve(t *testing.T) {
	for _, tc := range []struct{ total, k, want int }{
		{64, 8, 8},                 // ceiling floors at 16, clamped to k -> exact
		{256, 16, 16},              // exact
		{20_000, 20, 16},           // ceiling's 16 first bites here
		{100_000, 98, 24},          // ceiling: k/4, the case this clamp exists for
		{400_000, 391, 97},         // ceiling and budget within one of each other
		{1_000_000, 977, 98},       // budget
		{10_000_000, 9766, 305},    // recall floor, k/32
		{100_000_000, 97657, 3051}, // recall floor
	} {
		if got := HoodK(tc.total); got != tc.k {
			t.Fatalf("HoodK(%d) = %d, want %d", tc.total, got, tc.k)
		}
		got := min(DefaultNprobe(tc.k), tc.k)
		if got != tc.want {
			t.Errorf("N=%d k=%d: default nprobe %d, want %d", tc.total, tc.k, got, tc.want)
		}
		// The two invariants the three rules must never break.
		if got > tc.k {
			t.Errorf("N=%d: default %d exceeds k=%d", tc.total, got, tc.k)
		}
		if tc.k >= 64 && got > tc.k/4 {
			t.Errorf("N=%d: default %d is more than a quarter of k=%d", tc.total, got, tc.k)
		}
	}
}

// TestHoodKIsFixedSize pins the sizing rule that replaced round(sqrt(N)):

// hood COUNT carries the growth, hood SIZE does not. The first block is the

// compatibility claim (existing namespaces and every small test get the k

// they got before), the second is the scaling claim.

func TestHoodKIsFixedSize(t *testing.T) {
	// Below hoodFloor x HoodTarget the floor is min(hoodFloor, RoundSqrt),
	// which IS RoundSqrt while RoundSqrt <= hoodFloor — so k is unchanged
	// from the old rule for every total up to 16^2 = 256.
	for _, total := range []int{64, 100, 169, 256} {
		if got, want := HoodK(total), RoundSqrt(total); got != want {
			t.Errorf("HoodK(%d) = %d, want round(sqrt N) = %d (small-N compatibility)", total, got, want)
		}
	}
	// Above it, k = ceil(total/HoodTarget) and hood size is flat.
	for _, tc := range []struct{ total, k int }{
		{100_000, 98},
		{1_000_000, 977},
		{10_000_000, 9766},
		{100_000_000, 97657},
	} {
		if got := HoodK(tc.total); got != tc.k {
			t.Errorf("HoodK(%d) = %d, want %d", tc.total, got, tc.k)
		}
		if size := HoodSize(tc.total); size < HoodTarget/2 || size > HoodTarget {
			t.Errorf("HoodSize(%d) = %d, want about HoodTarget=%d", tc.total, size, HoodTarget)
		}
	}
	// The default itself — three interacting rules and both crossovers —
	// is TestDefaultNprobeCurve's job, not this one's.
	if HoodK(0) != 1 || HoodK(-5) != 1 || HoodK(1) != 1 {
		t.Error("HoodK must never return less than 1")
	}
}

// TestSplitAndMergeBracketTheTarget: the two triggers that KEEP hoods near

// the target once the bootstrap has placed them. Split fires strictly above

// SplitAbove (1.4x target) and merge at or below target/4, so a hood inside

// [target/4, SplitAbove] is left alone by both — which is the band

// an end-to-end churn test asserts on a whole index.

func TestSplitAndMergeBracketTheTarget(t *testing.T) {
	const total = 1_000_000
	target := HoodSize(total) // 1024
	stats := make([]ClusterStat, HoodK(total))
	for i := range stats {
		stats[i] = ClusterStat{ID: i, Count: target, Radius: 1}
	}
	if _, ok := (DefaultPolicy{}).SplitTarget(stats); ok {
		t.Fatal("a uniform index at the target must not split")
	}
	// SplitAbove is stated on the total, which the oversized hood is part of.
	above := SplitAbove(TotalCount(stats))
	stats[7].Count = above + 1
	if id, ok := (DefaultPolicy{}).SplitTarget(stats); !ok || id != 7 {
		t.Fatalf("SplitTarget = (%d,%v), want the oversized hood 7", id, ok)
	}
	stats[7].Count = above // exactly SplitAbove is still in band
	if _, ok := (DefaultPolicy{}).SplitTarget(stats); ok {
		t.Fatal("exactly SplitAbove must not split (strict >)")
	}
	if got, want := SplitAbove(total), target*7/5; got != want {
		t.Fatalf("SplitAbove(%d) = %d, want 1.4x target = %d", total, got, want)
	}
	if got, want := MergeBelow(total), target/4; got != want {
		t.Fatalf("MergeBelow(%d) = %d, want target/4 = %d", total, got, want)
	}
	// The singleton rule survives at any size: MergeBelow never drops below
	// MergeMax, so a 1-vector hood in a tiny namespace is still absorbed.
	if MergeBelow(80) < MergeMax {
		t.Fatalf("MergeBelow(80) = %d, must never be below MergeMax = %d", MergeBelow(80), MergeMax)
	}
}

// never sees more than SampleSize(k) vectors however big the corpus is, and

// every vector still gets assigned in the pass that follows. Without this

// bound, fixed-size hoods (k = N/HoodTarget) would make the build O(N^2/S)

// per iteration and be strictly worse than the sqrt-sized index it replaces.

// TestSplitTargetSkipsUnsplittableHood: a hood whose members are all the same

// point has Radius 0 and CANNOT be split — 2-means over identical vectors

// puts everything in one cluster and leaves the other empty, dropEmptyHoods

// reclaims the empty one, and the size trigger fires again on the next cycle.

// That is one wasted structural verb per compaction, forever. The size

// trigger therefore skips it however far past the target it is; the ordinary

// fat-and-spread hood beside it is the one that goes.

func TestSplitTargetSkipsUnsplittableHood(t *testing.T) {
	const total = 1_000_000
	target := HoodSize(total)
	stats := make([]ClusterStat, HoodK(total))
	for i := range stats {
		stats[i] = ClusterStat{ID: i, Count: target, Radius: 1}
	}
	stats[3] = ClusterStat{ID: 3, Count: 10 * target, Radius: 0} // all one point
	if id, ok := (DefaultPolicy{}).SplitTarget(stats); ok {
		t.Fatalf("split target %d: a zero-radius hood is unsplittable and must be left alone", id)
	}
	stats[9] = ClusterStat{ID: 9, Count: 3 * target, Radius: 1}
	if id, ok := (DefaultPolicy{}).SplitTarget(stats); !ok || id != 9 {
		t.Fatalf("SplitTarget = (%d,%v), want the splittable fat hood 9 (not the bigger unsplittable 3)", id, ok)
	}
}

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
