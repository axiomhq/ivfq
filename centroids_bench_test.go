package ivfq_test

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/axiomhq/ivfq"
)

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
				sink = ivfq.Nearest(cents, q)
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
					sink = ivfq.Nearest(cents, q)
				}
			})
			b.StopTimer()
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())*float64(runtime.GOMAXPROCS(0))/float64(b.N)/float64(k), "ns/centroid")
		})
	}
}

var sink int
