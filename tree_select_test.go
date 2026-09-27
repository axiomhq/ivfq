package ivfq

import (
	"math/rand"
	"slices"
	"testing"
)

// TestSelectNearestMatchesSort: the bounded selection is exactly the head
// of a full sort by (distance, id), for every probe including ones larger
// than the candidate set, with duplicate distances to exercise the tie.
func TestSelectNearestMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		n := rng.Intn(600)
		cands := make([]rankedNode, n)
		for i := range cands {
			cands[i] = rankedNode{id: rng.Intn(n + 1), d: float32(rng.Intn(50)) / 7}
		}
		probe := 1 + rng.Intn(40)
		want := slices.Clone(cands)
		slices.SortFunc(want, func(a, b rankedNode) int {
			switch {
			case a.before(b):
				return -1
			case b.before(a):
				return 1
			}
			return 0
		})
		want = want[:min(probe, len(want))]
		got := selectNearest(make([]rankedNode, 0, probe), cands, probe)
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d (n=%d probe=%d): got %v want %v", trial, n, probe, got, want)
		}
	}
}
