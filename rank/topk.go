package rank

import (
	"container/heap"
	"sort"
)

// Hit is one scored result.
type Hit struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}

// Select returns the k best hits — score desc, ties by ID asc for determinism —
// which is what a full sort followed by a truncation returns, at O(C log k)
// instead of O(C log C). k <= 0 yields no hits, so a negative caller k can
// never panic.
//
// At 10M rows the full sort was 51% of a warm query's wall time: 1,002,530
// candidates sorted, through a reflect swapper and a closure, to return 10.
func Select(hits []Hit, k int) []Hit {
	if k <= 0 {
		return hits[:0] // no hits, in the input's own shape (TestRRF pins the empty slice)
	}
	sel := NewTopK(k)
	for _, h := range hits {
		sel.Push(h.ID, h.Score)
	}
	return sel.Hits()
}

// TopK is Select as an accumulator: push candidates as they are scored and the
// slice holding all of them never has to exist. A query path scoring ~1M
// candidates would otherwise allocate ~24 MB of []Hit garbage per query on
// top of the sort.
//
// It keeps a k-sized min-heap under the INVERSE of the result order, so the
// root is the weakest hit kept and a candidate that cannot make the cut costs
// exactly one comparison. Zero value is unusable; call NewTopK.
type TopK struct {
	k int
	h hitHeap
}

// NewTopK returns a selector for the k best hits. k <= 0 accepts pushes and
// returns nothing, matching Select's guard.
func NewTopK(k int) *TopK { return &TopK{k: k} }

// Push offers one scored candidate. Order of pushes does not affect the
// result: worse fully determines the ranking of any two distinct hits.
func (t *TopK) Push(id string, score float32) {
	if t.k <= 0 {
		return
	}
	if len(t.h) < t.k {
		heap.Push(&t.h, Hit{ID: id, Score: score})
		return
	}
	if cand := (Hit{ID: id, Score: score}); worse(t.h[0], cand) {
		t.h[0] = cand
		heap.Fix(&t.h, 0)
	}
}

// Cutoff returns the weakest retained hit once the selector is full.
func (t *TopK) Cutoff() (Hit, bool) {
	if t.k <= 0 || len(t.h) < t.k {
		return Hit{}, false
	}
	return t.h[0], true
}

// Hits returns the selected hits in result order: score desc, ties by ID asc.
// Nothing selected yields nil, exactly as a truncating sort would. The sort
// here is over at most k elements, not over the candidates.
func (t *TopK) Hits() []Hit {
	out := []Hit(t.h)
	sort.Slice(out, func(i, j int) bool { return worse(out[j], out[i]) })
	return out
}

// worse is the result order inverted: a is worse than b when it sorts AFTER b
// — lower score, or the same score and a bigger id. One predicate for the
// heap and for the final k-element sort, so the two cannot drift.
func worse(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.ID > b.ID
}

// hitHeap is container/heap's contract over the selector's backing slice.
// Push boxes a Hit, but it only runs for the first k candidates — every one
// after that lands on Fix or on nothing at all.
type hitHeap []Hit

func (h hitHeap) Len() int           { return len(h) }
func (h hitHeap) Less(i, j int) bool { return worse(h[i], h[j]) }
func (h hitHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hitHeap) Push(x any)        { *h = append(*h, x.(Hit)) }
func (h *hitHeap) Pop() any          { old := *h; n := len(old) - 1; x := old[n]; *h = old[:n]; return x }
