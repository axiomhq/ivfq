package ivfq_test

import (
	"context"
	"math/rand"
	"runtime"
	"testing"

	"github.com/axiomhq/ivfq"
)

// benchCorpus is the scale run's corpus shape: n vectors of
// dims float32, 8 gaussian blobs, deterministic.
func benchCorpus(n, dims int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, 8)
	for i := range centers {
		centers[i] = make([]float32, dims)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64() * 10)
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[i%len(centers)]
		v := make([]float32, dims)
		for d := range v {
			v[d] = c[d] + float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// BenchmarkRun is the index build's inner loop at the two sizes the scale runs
// reports. GOMAXPROCS=1 reproduces the pre-2026-09-05 serial pass, so
//
//	go test -run x -bench Run/100k -benchtime 1x .
//	GOMAXPROCS=1 go test -run x -bench Run/100k -benchtime 1x .
//
// is the whole before/after. 1M takes ~19 minutes at GOMAXPROCS=1; -short
// skips it.
func BenchmarkRun(b *testing.B) {
	const iters = 10 // engine's kmeansIters
	cases := []struct {
		name       string
		n, dims, k int
	}{
		{"100k-d128-k316", 100_000, 128, 316},
		{"1M-d128-k1000", 1_000_000, 128, 1000},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			if c.n > 100_000 && testing.Short() {
				b.Skip("minutes-long build")
			}
			vecs := benchCorpus(c.n, c.dims, 31)
			b.ReportMetric(float64(runtime.GOMAXPROCS(0)), "procs")
			b.ResetTimer()
			for b.Loop() {
				ivfq.Run(context.Background(), vecs, c.k, iters, 1)
			}
		})
	}
}
