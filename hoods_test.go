package ivfq

import (
	"context"
	"math/rand"
	"testing"
)

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

// TestRunSampledBoundsTraining is the scaling rule as an assertion: Lloyd's
// never sees more than SampleSize(k) vectors however big the corpus is, and
// every vector still gets assigned in the pass that follows. Without this
// bound, fixed-size hoods (k = N/HoodTarget) would make the build O(N^2/S)
// per iteration and be strictly worse than the sqrt-sized index it replaces.
func TestRunSampledBoundsTraining(t *testing.T) {
	const n, dims, k = 50_000, 8, 64
	rng := rand.New(rand.NewSource(3))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dims)
		for d := range v {
			v[d] = float32(rng.NormFloat64()) + float32(i%16)
		}
		vecs[i] = v
	}
	ResetMaxTrainN()
	centroids, assign, _ := RunSampled(context.Background(), vecs, k, 10, 1)
	if got, want := MaxTrainN(), SampleSize(k); got != want {
		t.Fatalf("k-means trained on %d vectors, want exactly SampleSize(%d) = %d (n = %d)", got, k, want, n)
	}
	if len(centroids) != k || len(assign) != n {
		t.Fatalf("centroids=%d assign=%d, want %d and %d", len(centroids), len(assign), k, n)
	}
	// Every vector assigned, and assigned to its true nearest — the sample
	// picks the frame, the full pass places the corpus.
	for i := range assign {
		if assign[i] < 0 || assign[i] >= k {
			t.Fatalf("assign[%d] = %d out of range", i, assign[i])
		}
		if i%997 == 0 {
			if want := Nearest(centroids, vecs[i]); assign[i] != want {
				t.Fatalf("assign[%d] = %d, want nearest %d", i, assign[i], want)
			}
		}
	}
	// A corpus the sample already covers gets the full fit, unchanged.
	ResetMaxTrainN()
	small := vecs[:SampleSize(k)]
	got, _, _ := RunSampled(context.Background(), small, k, 10, 1)
	want, _, _ := Run(context.Background(), small, k, 10, 1)
	if MaxTrainN() != len(small) {
		t.Fatalf("sample >= corpus must train on all %d, trained on %d", len(small), MaxTrainN())
	}
	for i := range got {
		for d := range got[i] {
			if got[i][d] != want[i][d] {
				t.Fatalf("RunSampled != Run when the sample covers the corpus (centroid %d, dim %d)", i, d)
			}
		}
	}
}

func TestSampleSizeForBudget(t *testing.T) {
	if got := SampleSizeForBudget(1000, 128, 1<<20); got > (1<<20)/(128*4) {
		t.Fatalf("sample has %d vectors, exceeds byte budget", got)
	}
	if got := SampleSizeForBudget(1000, 128, 1); got != 1 {
		t.Fatalf("minimum sample = %d, want 1", got)
	}
}

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
