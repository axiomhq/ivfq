package ivfq

// The codec-independent half of the package's tests: the corpora and the
// recall arithmetic every measurement here shares, and Rerank itself. What
// the ONE codec guarantees — the round trip, the metadata it refuses, the
// estimator's bias, the bound's coverage, Query.Exact, and the two-pass
// recall the engine actually runs — is rabitq_test.go.

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

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
