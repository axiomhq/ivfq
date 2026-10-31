package rabitq

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/rank"
)

// The codec-independent half of the package's tests: the corpora and the
// recall arithmetic every measurement here shares, and Rerank itself. What
// the ONE codec guarantees — the round trip, the metadata it refuses, the
// estimator's bias, the bound's coverage, Query.Exact, and the two-pass
// recall the engine actually runs — is rabitq_test.go.

func corpus(n, d int) [][]float32 {
	r := rand.New(rand.NewSource(5))
	v := make([][]float32, n)
	for i := range v {
		v[i] = make([]float32, d)
		for j := range v[i] {
			v[i][j] = float32(r.NormFloat64())
		}
	}
	return v
}

func top(v [][]float32, q []float32, score func(int) float32, k int) []int {
	ids := make([]int, len(v))
	for i := range ids {
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool { a, b := score(ids[i]), score(ids[j]); return a > b || a == b && ids[i] < ids[j] })
	return ids[:k]
}

func recall(a, b []int) float64 {
	m := map[int]bool{}
	for _, x := range a {
		m[x] = true
	}
	n := 0
	for _, x := range b {
		if m[x] {
			n++
		}
	}
	return float64(n) / float64(len(b))
}

func clusteredCorpus(n, clusters, dims int) [][]float32 {
	r := rand.New(rand.NewSource(int64(clusters + dims)))
	centers := corpus(clusters, dims)
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, dims)
		for d := range out[i] {
			out[i][d] = centers[i%clusters][d] + float32(r.NormFloat64())*.05
		}
	}
	return out
}

func TestRerankSupersetProperty(t *testing.T) {
	scores := []float32{1, 9, 4, 7, 2}
	got := Rerank([]int{4, 3, 2, 1, 0}, func(i int) float32 { return scores[i] }, 2)
	if !reflect.DeepEqual(got, []int{1, 3}) {
		t.Fatal(got)
	}
}

// Rerank scores candidate IDs exactly and returns the best k, with lower IDs

// winning equal scores. exactDistance must return a higher-is-better score.

func Rerank(candidates []int, exactDistance func(int) float32, k int) []int {
	r := make([]ranked, len(candidates))
	for i, id := range candidates {
		r[i] = ranked{id, exactDistance(id)}
	}
	sort.Slice(r, func(i, j int) bool { return r[i].score > r[j].score || r[i].score == r[j].score && r[i].id < r[j].id })
	k = min(max(k, 0), len(r))
	out := make([]int, k)
	for i := range out {
		out[i] = r[i].id
	}
	return out
}

// A scorer rebound to another column in the same frame (Scorer.With, the

// tier's per-cluster binding) scores that column's rows exactly as a

// scorer bound to it directly, whichever representation either column has

// and however many rows it holds.

func TestReboundScorerReadsTheNewColumn(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		opts := Options{Metric: metric, Rotation: NewRotation(11, 129)}
		centroid := corpus(1, 129)[0]
		small, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if small, err = AppendRows(small, opts.Rotation, corpus(5, 129)); err != nil {
			t.Fatal(err)
		}
		big, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if big, err = AppendRows(big, opts.Rotation, corpus(64, 129)[5:]); err != nil {
			t.Fatal(err)
		}
		data, err := big.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		bigBorrowed, err := UnmarshalBinaryBorrowed(data)
		if err != nil {
			t.Fatal(err)
		}
		q := NewQuery(corpus(2, 129)[1], metric, opts.Rotation)
		for _, target := range []Quantizer{big, bigBorrowed} {
			if !small.SameFrame(&target) {
				t.Fatalf("%s: columns are not in one frame", metric)
			}
			rebound := small.Scorer(q).With(&target)
			direct := target.Scorer(q)
			for row := 0; row < target.Rows(); row++ {
				gs, gb := rebound.ScoreAndBound(row)
				ws, wb := direct.ScoreAndBound(row)
				if gs != ws || gb != wb {
					t.Fatalf("%s row %d: rebound (%v,%v), direct (%v,%v)", metric, row, gs, gb, ws, wb)
				}
			}
		}
	}
}

// Exercise lazy scoring across the existing packed/borrowed/global/ordinary
// matrix. The independent row estimator supplies the bound oracle.
func checkLazyBounds(t *testing.T, name string, s Scorer, rows int) {
	t.Helper()
	scores, bounds := make([]float32, rows), make([]float32, rows)
	hint := s.ScoreAllLazy(scores)
	cuts := []float32{float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.NaN()), 0, math.Float32frombits(1 << 31)}
	if rows > 0 {
		_, bound := s.ScoreAndBound(rows / 2)
		cuts = append(cuts, bound, math.Nextafter32(bound, float32(math.Inf(-1))), math.Nextafter32(bound, float32(math.Inf(1))))
	}
	for _, cutoff := range cuts {
		s.RefineBounds(scores, bounds, cutoff, hint)
		for row := range scores {
			wantScore, wantBound := s.ScoreAndBound(row)
			if math.Float32bits(scores[row]) != math.Float32bits(wantScore) {
				t.Fatalf("%s row %d: lazy score %g, want %g", name, row, scores[row], wantScore)
			}
			if (bounds[row] < cutoff) != (wantBound < cutoff) || bounds[row] < wantBound {
				t.Fatalf("%s row %d: cutoff %g, lazy bound %g, want %g", name, row, cutoff, bounds[row], wantBound)
			}
			if !(bounds[row] < cutoff) && math.Float32bits(bounds[row]) != math.Float32bits(wantBound) {
				t.Fatalf("%s row %d: survivor bound bits changed", name, row)
			}
		}
	}
}

func TestRefineBoundsAfterRebinding(t *testing.T) {
	c, rot := globalColumn(t, corpus(35, 128), ivfq.L2)
	var s Scorer
	q := NewQuery(corpus(1, 128)[0], ivfq.L2, rot).Global(rot)
	if !s.BindGlobal(&c, q) {
		t.Fatal("global bind failed")
	}
	scores, bounds := make([]float32, c.Rows()), make([]float32, c.Rows())
	hint := s.ScoreAllLazy(scores)
	// Reuse the metadata hint, but rescore for each new query and confidence.
	for _, sigmas := range []float64{0.01, 3.2, math.Inf(1)} {
		t.Run(fmt.Sprint(sigmas), func(t *testing.T) {
			q := NewQuery(offCentre(1, 128, 3)[0], ivfq.L2, rot).Global(rot)
			q.Sigmas = sigmas
			if !s.BindGlobal(&c, q) {
				t.Fatal("global rebind failed")
			}
			s.ScoreAll(scores, nil)
			for _, cutoff := range []float32{-1e6, 0} {
				s.RefineBounds(scores, bounds, cutoff, hint)
				for row := range bounds {
					_, want := s.ScoreAndBound(row)
					if (bounds[row] < cutoff) != (want < cutoff) || bounds[row] < want {
						t.Fatalf("row %d: cutoff %g, rebound %g, want %g", row, cutoff, bounds[row], want)
					}
				}
			}
			checkLazyBounds(t, "rebound", s, c.Rows())
		})
	}
	if !s.BindGlobal(&c, q) {
		t.Fatal("original rebind failed")
	}
	other, err := c.Select([]int{34, 0, 17})
	if err != nil {
		t.Fatal(err)
	}
	other.PackFastScanFrame(RotateCentroid(other.Code.Centroid, rot))
	s.Use(&other)
	s.ScoreAll(scores, nil)
	for _, h := range []BoundHint{hint, {}} {
		if n := s.RefineBounds(scores, bounds, 0, h); n != other.Rows() {
			t.Fatalf("foreign/zero hint evaluated %d bounds, want %d", n, other.Rows())
		}
		for row := range other.Rows() {
			_, want := s.ScoreAndBound(row)
			if math.Float32bits(bounds[row]) != math.Float32bits(want) {
				t.Fatalf("foreign/zero hint changed bound at row %d", row)
			}
		}
	}
}

func ExampleScorer_RefineBounds() {
	rows := [][]float32{{0}, {1}, {8}, {9}, {10}}
	rot := NewRotation(11, 1)
	col, err := Quantize(rows, Options{Metric: ivfq.L2, Rotation: rot})
	if err != nil {
		panic(err)
	}
	col.PackFastScanFrame(RotateCentroid(col.Code.Centroid, rot))
	query := []float32{0.9}
	var s Scorer
	if !s.BindGlobal(&col, NewQuery(query, ivfq.L2, rot).Global(rot)) {
		panic("global bind failed")
	}
	scores, bounds := make([]float32, col.Rows()), make([]float32, col.Rows())
	hint := s.ScoreAllLazy(scores)
	shortlist := rank.NewTopK(2)
	for row, score := range scores {
		shortlist.Push(fmt.Sprint(row), score)
	}
	top := rank.NewTopK(1)
	seen := make(map[string]bool)
	for _, hit := range shortlist.Hits() {
		var row int
		fmt.Sscan(hit.ID, &row)
		top.Push(hit.ID, ivfq.Score(ivfq.L2, query, rows[row]))
		seen[hit.ID] = true
	}
	cutoff, _ := top.Cutoff() // the exact top-k is full before pruning
	s.RefineBounds(scores, bounds, cutoff.Score, hint)
	for row, bound := range bounds {
		id := fmt.Sprint(row)
		if bound < cutoff.Score || seen[id] {
			continue
		}
		top.Push(id, ivfq.Score(ivfq.L2, query, rows[row]))
		cutoff, _ = top.Cutoff() // only rises; retain original row order
	}
	fmt.Println(top.Hits()[0].ID)
	// Output: 1
}
