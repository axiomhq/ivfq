package vsweaviate

// 2-means for cluster splits: ivfq/kmeans (k-means++ seed, Lloyd, no size
// constraint) against weaviate's KMeans.FitBalanced (random init, Lloyd with a
// size floor of ceil(n/4) per side), configured as KMeansEncoder.FitBalanced
// configures it.

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/axiomhq/ivfq/internal/simd"
	"github.com/axiomhq/ivfq/kmeans"
	wk "github.com/weaviate/weaviate/adapters/repos/db/vector/kmeans"
)

// splitData: two gaussian blobs, frac of the points in the first, separated by
// sep sigmas along a random direction. frac 0.5 balanced, 0.9 skewed, and
// sep 0 is one blob (an unsplittable cluster: the case a balance floor exists for).
func splitData(n, d int, frac, sep float64, r *rand.Rand) [][]float32 {
	dir := make([]float32, d)
	for j := range dir {
		dir[j] = float32(r.NormFloat64())
	}
	simd.Normalize(dir)
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, d)
		off := float32(0)
		if float64(i) >= frac*float64(n) {
			off = float32(sep)
		}
		for j := range out[i] {
			out[i][j] = float32(r.NormFloat64()) + off*dir[j]
		}
	}
	r.Shuffle(n, func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

type splitter struct {
	name string
	fit  func(data [][]float32) (centres [][]float32, assign []int)
}

func splitters(d int) []splitter {
	wv := func(balanced bool) func([][]float32) ([][]float32, []int) {
		return func(data [][]float32) ([][]float32, []int) {
			km := wk.New(2, d, 0)
			km.DeltaThreshold = 0.01
			km.IterationThreshold = 10
			km.Initialization = wk.RandomInitialization
			km.Assignment = wk.GraphPruning
			assign := make([]int, len(data))
			if balanced {
				a, err := km.FitBalanced(data)
				if err != nil {
					panic(err)
				}
				for i, x := range a {
					assign[i] = int(x)
				}
				return km.Centers, assign
			}
			if err := km.Fit(data); err != nil {
				panic(err)
			}
			for i, x := range data {
				assign[i] = kmeans.Nearest(km.Centers, x)
			}
			return km.Centers, assign
		}
	}
	return []splitter{
		{"ivfq Fit(K=2,Iters=8)", func(data [][]float32) ([][]float32, []int) {
			c, a, err := kmeans.Config{K: 2, Iters: kmeans.SplitIters, Seed: 1}.Fit(context.Background(), data)
			if err != nil {
				panic(err)
			}
			return c, a
		}},
		{"ivfq FitBalanced", func(data [][]float32) ([][]float32, []int) {
			c, a, err := kmeans.Config{K: 2, Iters: kmeans.SplitIters, Seed: 1}.FitBalanced(context.Background(), data)
			if err != nil {
				panic(err)
			}
			return c, a
		}},
		{"wv Fit(K=2)", wv(false)},
		{"wv FitBalanced", wv(true)},
	}
}

func wcss(data [][]float32, c [][]float32, a []int) (w float64, minFrac float64) {
	var n [2]int
	for i, x := range data {
		w += float64(simd.L2Sq(x, c[a[i]]))
		n[a[i]]++
	}
	return w, float64(min(n[0], n[1])) / float64(len(data))
}

type scenario struct {
	name      string
	frac, sep float64
}

var scenarios = []scenario{
	{"balanced 50/50, sep 20", 0.5, 20},
	{"skewed 90/10, sep 20", 0.9, 20},
	{"skewed 98/2, sep 20", 0.98, 20},
	{"one blob (no structure)", 0.5, 0},
}

// TestSplitQuality logs WCSS (relative to ivfq's) and the smaller side's share
// of the points. Run with -v.
func TestSplitQuality(t *testing.T) {
	const n, d = 4096, 128
	t.Logf("n=%d d=%d  (WCSS relative to ivfq, smaller side's share of points; lower WCSS is better)", n, d)
	for _, sc := range scenarios {
		data := splitData(n, d, sc.frac, sc.sep, rand.New(rand.NewPCG(3, 4)))
		var base float64
		for _, s := range splitters(d) {
			c, a := s.fit(data)
			w, frac := wcss(data, c, a)
			if base == 0 {
				base = w
			}
			t.Logf("%-26s %-24s wcss %6.3fx  min side %5.1f%%", sc.name, s.name, w/base, 100*frac)
		}
	}
}

func BenchmarkSplit(b *testing.B) {
	for _, d := range []int{128, 768} {
		for _, n := range []int{1024, 4096} {
			data := splitData(n, d, 0.5, 20, rand.New(rand.NewPCG(3, 4)))
			for _, s := range splitters(d) {
				b.Run(fmt.Sprintf("d=%d/n=%d/%s", d, n, s.name), func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						s.fit(data)
					}
				})
			}
		}
	}
}
