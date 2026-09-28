package kmeans_test

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/axiomhq/ivfq/kmeans"
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
				kmeans.Run(context.Background(), vecs, c.k, iters, 1)
			}
		})
	}
}

// BenchmarkCentroidScan is one query's centroid selection: the full L2 scan
// over k centroids that both a query (ranking hoods for nprobe) and the
// build's assignment pass (once per vector) pay. Under fixed-size hoods k
// grows as N/HoodTarget, so this is the term that decides whether a second,
// coarse centroid level is needed — parity.md's gate, and backlog 31's
// measurement of the 1.49x the build paid when the set left L2 at 10M.
//
// The k values are the sizing rule's answers: 977 at 1M (512 KB, fits L2),
// 9,766 at 10M (5.0 MB, does not), 97,657 at 100M (50 MB).
//
//	go test -run x -bench CentroidScan .
func BenchmarkCentroidScan(b *testing.B) {
	const dims = 128
	for _, k := range []int{977, 9766, 97657} {
		b.Run(fmt.Sprintf("k=%d", k), func(b *testing.B) {
			cents := benchCorpus(k, dims, 7)
			q := benchCorpus(1, dims, 9)[0]
			b.ReportMetric(float64(k*dims*4)/1024, "KiB/centroid-set")
			b.ResetTimer()
			for b.Loop() {
				sink = kmeans.Nearest(cents, q)
			}
			b.StopTimer()
			// ns per centroid is the number that has to stay flat; when it
			// does not, the set has fallen out of cache.
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(k), "ns/centroid")
		})
		// The same scan the way the BUILD runs it: every core streaming the
		// whole centroid set at once, which is what shares (and thrashes) the
		// last-level cache. backlog 31 measured 20.4 -> 30.4 ns per distance
		// call between 1M and 10M and attributed it to the set leaving L2;
		// the serial arm above is flat, so if the tax is real it is here.
		b.Run(fmt.Sprintf("k=%d/parallel", k), func(b *testing.B) {
			cents := benchCorpus(k, dims, 7)
			qs := benchCorpus(64, dims, 9)
			b.ResetTimer()
			var i atomic.Int64
			b.RunParallel(func(pb *testing.PB) {
				q := qs[int(i.Add(1))%len(qs)]
				for pb.Next() {
					sink = kmeans.Nearest(cents, q)
				}
			})
			b.StopTimer()
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())*float64(runtime.GOMAXPROCS(0))/float64(b.N)/float64(k), "ns/centroid")
		})
	}
}

var sink int
