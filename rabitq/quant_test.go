package rabitq

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/axiomhq/ivfq"
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
