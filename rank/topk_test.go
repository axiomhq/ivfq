package rank

import (
	"math/rand"
	"sort"
	"strconv"
	"testing"
)

func TestTopKTieBreak(t *testing.T) {
	hits := Select([]Hit{{"b", 1}, {"a", 1}, {"c", 2}}, 2)
	if hits[0].ID != "c" || hits[1].ID != "a" {
		t.Fatalf("tie-break not deterministic: %+v", hits)
	}
}

// sortedTopK is what Select replaced with a bounded selection: a full
// comparison sort of every candidate, then a truncation. It is the reference
// the property test below holds the heap to — same order, same length, for
// every k including the degenerate ones.
func sortedTopK(hits []Hit, k int) []Hit {
	if k < 0 {
		k = 0
	}
	out := append([]Hit(nil), hits...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID
	})
	if len(out) > k {
		out = out[:k]
	}
	return out
}

func TestTopKMatchesSortedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := range 200 {
		n := rng.Intn(40)
		hits := make([]Hit, n)
		for i := range hits {
			// Few distinct scores and few distinct ids on purpose: ties on
			// score (broken by id ascending) and outright duplicates are what
			// a bounded heap gets wrong, and at these widths they are certain.
			hits[i] = Hit{ID: string(rune('a' + rng.Intn(12))), Score: float32(rng.Intn(8))}
		}
		for _, k := range []int{0, 1, n - 1, n, n + 5, -3} {
			got, want := Select(append([]Hit(nil), hits...), k), sortedTopK(hits, k)
			if len(got) != len(want) {
				t.Fatalf("trial %d k=%d: len %d, want %d\n got %+v\nwant %+v", trial, k, len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("trial %d k=%d: hit %d = %+v, want %+v\n got %+v\nwant %+v", trial, k, i, got[i], want[i], got, want)
				}
			}
		}
	}
}

// BenchmarkTopK is a query path's selection step in isolation: one
// million candidates, ten winners.
func BenchmarkTopK(b *testing.B) {
	rng := rand.New(rand.NewSource(11))
	hits := make([]Hit, 1_000_000)
	for i := range hits {
		hits[i] = Hit{ID: strconv.Itoa(i), Score: rng.Float32()}
	}
	scratch := make([]Hit, len(hits))
	b.ReportAllocs()
	for b.Loop() {
		copy(scratch, hits) // Select may reorder its input; every iteration gets the same one
		Select(scratch, 10)
	}
}

func TestTopKCutoff(t *testing.T) {
	top := NewTopK(2)
	if _, ok := top.Cutoff(); ok {
		t.Fatal("partial selector has cutoff")
	}
	top.Push("b", 1)
	top.Push("a", 1)
	if got, ok := top.Cutoff(); !ok || got.ID != "b" || got.Score != 1 {
		t.Fatalf("cutoff = %+v, %v", got, ok)
	}
}
