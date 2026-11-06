package rabitq

import (
	"context"
	"math/rand"
	"slices"
	"sort"
	"sync"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/x448/float16"
)

// blockCorpus is n leaf centroids around 32 shared centers, with unit noise.
func blockCorpus(n, dims int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, 32)
	for i := range centers {
		centers[i] = make([]float32, dims)
		for j := range centers[i] {
			centers[i][j] = float32(3 * rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		out[i] = make([]float32, dims)
		for j := range out[i] {
			out[i][j] = c[j] + float32(rng.NormFloat64())
		}
	}
	return out
}

// A Tree of RaBitQ blocks routes by estimates, so it is held to recall,
// not equality: the exact top 10 by squared distance over the raw leaf
// centroids, computed here in float64, must mostly fall within the tree's
// 40 nearest. The same tree answers identically from concurrent searches.
func TestBlockTreeRecallAgainstRawCentroids(t *testing.T) {
	const dims, topk, probe = 64, 10, 40
	ctx := context.Background()
	leaves := blockCorpus(1024, dims, 1)
	queries := blockCorpus(100, dims, 2)
	rot := NewRotation(Seed("block", "test"), dims)
	tree, err := ivfq.BuildTree(ctx, leaves, 16, func(rows [][]float32) (Block, error) { return NewBlock(rows, rot) })
	if err != nil {
		t.Fatal(err)
	}
	covered := 0
	got := make([][]int, len(queries))
	for i, q := range queries {
		ids := make([]int, len(leaves))
		d := make([]float64, len(leaves))
		for id, v := range leaves {
			ids[id] = id
			for j := range v {
				x := float64(q[j]) - float64(v[j])
				d[id] += x * x
			}
		}
		sort.SliceStable(ids, func(a, b int) bool { return d[ids[a]] < d[ids[b]] })
		got[i] = tree.Nearest(q, 64, probe)
		for _, id := range ids[:topk] {
			if slices.Contains(got[i], id) {
				covered++
			}
		}
	}
	// Measured 0.97 here (0.94 to 0.999 across synthetic and Big ANN
	// centroid sets); a sign, scale or id mistake scores near zero.
	if r := float64(covered) / float64(topk*len(queries)); r < 0.9 {
		t.Fatalf("exact top %d within the tree's %d: recall %.3f", topk, probe, r)
	}

	half, err := ivfq.BuildTree(ctx, leaves, 16, ivfq.NewDense[float16.Float16])
	if err != nil {
		t.Fatal(err)
	}
	halfWant := make([][]int, len(queries))
	for i, q := range queries {
		halfWant[i] = half.Nearest(q, 8, topk)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for i, q := range queries {
				if !slices.Equal(tree.Nearest(q, 64, probe), got[i]) || !slices.Equal(half.Nearest(q, 8, topk), halfWant[i]) {
					t.Error("a concurrent search disagreed")
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestBlockShape(t *testing.T) {
	rot := NewRotation(3, 4)
	rows := [][]float32{{1, 2, 3, 4}, {-1, 0, 2, 5}, {0, 0, 0, 0}}
	b, err := NewBlock(rows, rot)
	if err != nil {
		t.Fatal(err)
	}
	if b.Len() != 3 || b.Dims() != 4 {
		t.Fatalf("shape %d x %d", b.Len(), b.Dims())
	}
	q := []float32{0.5, -1, 2, 3}
	scores := []float32{7, 7, 7, 7, 7}
	b.ScoreAll(q, scores)
	if scores[3] != 7 || scores[4] != 7 || !slices.Equal(q, []float32{0.5, -1, 2, 3}) {
		t.Fatalf("ScoreAll wrote past Len or into the query: %v %v", scores, q)
	}
	if _, err := NewBlock([][]float32{{1, 2, 3}}, rot); err == nil {
		t.Fatal("a row of the wrong width was accepted")
	}
	if _, err := NewBlock(rows, nil); err == nil {
		t.Fatal("a nil rotation was accepted")
	}
}
