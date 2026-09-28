package kmeans

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"sync/atomic"
	"testing"
)

var ctx = context.Background()

// blobs returns n vectors around each of the given centers (deterministic).

func blobs(seed int64, n int, centers ...[]float32) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	var out [][]float32
	for _, c := range centers {
		for i := 0; i < n; i++ {
			v := make([]float32, len(c))
			for d := range c {
				v[d] = c[d] + float32(rng.NormFloat64())*0.05
			}
			out = append(out, v)
		}
	}
	return out
}

func TestRunRecoversBlobs(t *testing.T) {
	vecs := blobs(1, 50, []float32{0, 0}, []float32{10, 0}, []float32{0, 10})
	centroids, assign, _ := Run(ctx, vecs, 3, 10, 42)
	if len(centroids) != 3 || len(assign) != 150 {
		t.Fatalf("shape: %d centroids, %d assigns", len(centroids), len(assign))
	}
	// Purity: every vector in a blob shares its blob-mates' cluster.
	for b := 0; b < 3; b++ {
		want := assign[b*50]
		for i := b * 50; i < (b+1)*50; i++ {
			if assign[i] != want {
				t.Fatalf("blob %d impure at %d: %d vs %d", b, i, assign[i], want)
			}
		}
	}
	// Assignment is consistent with Nearest.
	for i, v := range vecs {
		if Nearest(centroids, v) != assign[i] {
			t.Fatalf("assign[%d] disagrees with Nearest", i)
		}
	}
}

func TestDeterminism(t *testing.T) {
	vecs := blobs(2, 30, []float32{0, 0}, []float32{5, 5})
	c1, a1, _ := Run(ctx, vecs, 2, 10, 7)
	c2, a2, _ := Run(ctx, vecs, 2, 10, 7)
	if !reflect.DeepEqual(c1, c2) || !reflect.DeepEqual(a1, a2) {
		t.Fatal("same seed, different output")
	}
}

func TestClampAndEdge(t *testing.T) {
	vecs := [][]float32{{1, 1}, {2, 2}}
	c, a, _ := Run(ctx, vecs, 10, 5, 1) // k > n clamps to n
	if len(c) != 2 || len(a) != 2 {
		t.Fatalf("clamp: %d centroids", len(c))
	}
	if c2, a2, _ := Run(ctx, nil, 3, 5, 1); c2 != nil || a2 != nil {
		t.Fatal("empty input should return nils")
	}
}

func TestReseedDistinct(t *testing.T) {
	// Duplicate-heavy input with k above the natural cluster count forces
	// simultaneous empty clusters: each must reseed to a DISTINCT vector
	// (review fix — un-claimed reseeds all cloned one vector and the
	// duplicates were a stable fixed point).
	vecs := [][]float32{{0, 0}, {0, 0}, {100, 100}, {100, 100}, {50, 0}, {0, 50}}
	centroids, _, _ := Run(ctx, vecs, 4, 8, 0)
	assertDistinct(t, centroids)
}

func TestReseedNoSteal(t *testing.T) {
	// Round-2 review found an assign-mutating claim could steal a live
	// cluster's sole member, minting a duplicate at low iteration budgets.
	// ALL-DISTINCT input isolates reseed dynamics: initial-permutation
	// collisions (a pre-existing, data-limited degeneracy on duplicate
	// inputs) are impossible by construction, so ANY output duplicate
	// indicts the reseed path. 50 deterministic seeds sweep the space.
	vecs := [][]float32{
		{0, 2}, {3, 4}, {0, 1}, {3, 1}, {0, 1.01},
		{3, 4.01}, {3, 1.01}, {2, 1}, {0, 1.02}, {0, 1.03},
	}
	for seed := int64(0); seed < 50; seed++ {
		centroids, _, _ := Run(ctx, vecs, 5, 3, seed)
		assertDistinct(t, centroids)
	}
}

func TestSphericalUnitCentroids(t *testing.T) {
	vecs := blobs(6, 40, []float32{3, 4}, []float32{-4, 3}) // non-unit inputs
	for i := range vecs {                                   // normalize inputs like a cosine corpus
		var n float64
		for _, x := range vecs[i] {
			n += float64(x) * float64(x)
		}
		s := float32(1 / math.Sqrt(n))
		for d := range vecs[i] {
			vecs[i][d] *= s
		}
	}
	centroids, _, _ := RunSpherical(ctx, vecs, 2, 10, 3)
	for i, c := range centroids {
		var n float64
		for _, x := range c {
			n += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(n)-1) > 1e-3 {
			t.Fatalf("centroid %d not unit: |c|=%f", i, math.Sqrt(n))
		}
	}
	c2, _, _ := RunSpherical(ctx, vecs, 2, 10, 3)
	if !reflect.DeepEqual(centroids, c2) {
		t.Fatal("spherical determinism broken")
	}
}

// TestWorkerCountInvariant: the assignment pass fans out over GOMAXPROCS, so

// the guarantee a stored index depends on is that the fan-out cannot change

// the answer — same seed, same corpus, same centroids and assignment on 1

// worker and on 16. (Across MACHINES or kernels it is not guaranteed and

// never was: L2Sq's summation order is part of the input.)

func TestWorkerCountInvariant(t *testing.T) {
	vecs := blobs(9, 400, []float32{0, 0}, []float32{7, 1}, []float32{-6, 4}, []float32{2, -9}, []float32{9, 9})
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1)) // serial path, restored after
	c1, a1, _ := Run(ctx, vecs, 64, 10, 11)         // n*k = 128k, above ivf's parallel floor
	runtime.GOMAXPROCS(16)
	c2, a2, _ := Run(ctx, vecs, 64, 10, 11)
	if !reflect.DeepEqual(c1, c2) || !reflect.DeepEqual(a1, a2) {
		t.Fatal("assignment fan-out changed the result: 1 worker and 16 disagree")
	}
}

// TestRunCancelled: a cancelled context returns its error and no result,

// on both the full fit and the sampled fit's assignment pass.

func TestRunCancelled(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	vecs := blobs(4, 40, []float32{0, 0}, []float32{5, 5})
	if c, a, err := Run(cancelled, vecs, 2, 10, 1); err != context.Canceled || c != nil || a != nil {
		t.Fatalf("Run on cancelled ctx = (%v, %v, %v)", c, a, err)
	}
	big := blobs(5, SampleSize(2)+1, []float32{0, 0}, []float32{5, 5})
	if c, a, err := (Config{K: 2, Iters: 10, Seed: 1}).Fit(cancelled, big); err != context.Canceled || c != nil || a != nil {
		t.Fatalf("RunSampled on cancelled ctx = (%v, %v, %v)", c, a, err)
	}
}

// TestRunSampledScratchIsBounded: sampling a corpus must not allocate in

// proportion to it. The one O(n) allocation RunSampled owns is the

// assignment array (8n bytes); the sample draw is O(SampleSize), so total

// allocation stays well under twice the assignment array.

func TestRunSampledScratchIsBounded(t *testing.T) {
	const n, dims, k = 1 << 20, 4, 2
	vecs := make([][]float32, n)
	flat := make([]float32, n*dims)
	for i := range vecs {
		vecs[i] = flat[i*dims : (i+1)*dims]
		vecs[i][0] = float32(i % 7)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, assign, err := (Config{K: k, Iters: 2, Seed: 1}).Fit(ctx, vecs); err != nil || len(assign) != n {
		t.Fatalf("RunSampled = (%d assigned, %v)", len(assign), err)
	}
	runtime.ReadMemStats(&after)
	if got, limit := after.TotalAlloc-before.TotalAlloc, uint64(8*n+8*n/2); got > limit {
		t.Fatalf("RunSampled allocated %d bytes for n=%d, limit %d: scratch is not bounded by the sample", got, n, limit)
	}
}

// cancelAfterPolls is a context whose Err reports cancellation from its

// at-th poll onward, so a test can cancel at an exact point inside a run

// and count how many polls follow — where k-means notices depends only on

// where it polls, never on a clock or the scheduler.

type cancelAfterPolls struct {
	context.Context
	at    int64
	polls atomic.Int64
}

func (c *cancelAfterPolls) Err() error {
	if c.polls.Add(1) >= c.at {
		return context.Canceled
	}
	return nil
}

// sampledCorpus is a corpus the sample does not cover (n = 4 x SampleSize

// at k=2) whose n*k assignment work stays under ivf's parallel floor, so

// every pass is serial and the poll sequence is fixed.

func sampledCorpus() ([][]float32, int) {
	const k = 2
	n := 4 * SampleSize(k)
	rng := rand.New(rand.NewSource(5))
	vecs := make([][]float32, n)
	flat := make([]float32, n*4)
	for i := range vecs {
		vecs[i] = flat[i*4 : (i+1)*4]
		for d := range vecs[i] {
			vecs[i][d] = float32(rng.NormFloat64()) + float32(i%16)
		}
	}
	return vecs, k
}

// TestRunSampledCancelDuringSampling: cancellation landing while the

// training sample is being drawn returns at that poll — before Lloyd's is

// ever handed the sample, and without a result.

func TestRunSampledCancelDuringSampling(t *testing.T) {
	vecs, k := sampledCorpus()
	pc := &cancelAfterPolls{Context: ctx, at: 2}
	c, a, err := (Config{K: k, Iters: 10, Seed: 1}).Fit(pc, vecs)
	if err != context.Canceled || c != nil || a != nil {
		t.Fatalf("RunSampled cancelled while sampling = (%v, %d assigned, %v)", c, len(a), err)
	}
	if got := pc.polls.Load(); got != 2 {
		t.Fatalf("%d polls after cancellation on the 2nd: sampling ran on", got-2)
	}
}

// TestRunSampledCancelMidPass: cancellation landing inside the serial

// full-corpus assignment pass is honored at the next chunk, not after the

// whole pass. An uncancelled run fixes the poll sequence; the pass is the

// last ceil(n/256) chunk polls before the final check, and cancelling in

// its middle must cost exactly one more poll (that final check).

func TestRunSampledCancelMidPass(t *testing.T) {
	vecs, k := sampledCorpus()
	full := &cancelAfterPolls{Context: ctx, at: math.MaxInt64}
	if _, _, err := (Config{K: k, Iters: 10, Seed: 1}).Fit(full, vecs); err != nil {
		t.Fatal(err)
	}
	chunks := int64((len(vecs) + 255) / 256)
	at := full.polls.Load() - chunks/2
	pc := &cancelAfterPolls{Context: ctx, at: at}
	c, a, err := (Config{K: k, Iters: 10, Seed: 1}).Fit(pc, vecs)
	if err != context.Canceled || c != nil || a != nil {
		t.Fatalf("RunSampled cancelled mid-pass = (%v, %d assigned, %v)", c, len(a), err)
	}
	if got := pc.polls.Load(); got != at+1 {
		t.Fatalf("%d polls after cancellation on poll %d of %d: the pass ran to completion", got-at, at, full.polls.Load())
	}
}

func assertDistinct(t *testing.T, centroids [][]float32) {
	t.Helper()
	seen := map[string]bool{}
	for _, c := range centroids {
		key := fmt.Sprintf("%v", c)
		if seen[key] {
			t.Fatalf("duplicate centroid %s in %v", key, centroids)
		}
		seen[key] = true
	}
}

// Test-only helpers: nothing outside this package's tests calls them.

// Run clusters vecs into k groups (k clamped to [1, len(vecs)]) with a fixed

// iteration cap. Deterministic for a given seed — including across worker

// counts, see assignAll. Empty clusters are reseeded from the vector

// farthest from its assigned centroid. A cancelled ctx returns its error

// and no result: checked at every seeding step, every Lloyd iteration,

// and every assignment chunk.

func Run(ctx context.Context, vecs [][]float32, k, iters int, seed int64) ([][]float32, []int, error) {
	return run(ctx, vecs, k, iters, seed, false)
}

// RunSpherical is Run for unit-vector corpora (cosine namespaces):

// centroids are re-normalized to unit length after every mean

// recomputation, keeping centroid geometry on the sphere. Plain L2

// against unit centroids is then monotone in cosine, so assignment and

// query-time selection stay L2 everywhere (spherical k-means). A

// zero-norm mean (degenerate antipodal cluster) is left unnormalized.

func RunSpherical(ctx context.Context, vecs [][]float32, k, iters int, seed int64) ([][]float32, []int, error) {
	return run(ctx, vecs, k, iters, seed, true)
}

// TestRunSampledBoundsTraining is the scaling rule as an assertion: Lloyd's

func TestRunSampledBoundsTraining(t *testing.T) {
	const n, dims, k = 50_000, 8, 64
	rng := rand.New(rand.NewSource(3))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dims)
		for d := range v {
			v[d] = float32(rng.NormFloat64()) + float32(i%16)
		}
		vecs[i] = v
	}
	centroids, assign, _ := (Config{K: k, Iters: 10, Seed: 1}).Fit(context.Background(), vecs)
	// RunSampled hands Lloyd's exactly sample(SampleSize(k)) and nothing
	// else: the centroids are the fit over that sample, bit for bit.
	trained, err := sample(context.Background(), vecs, SampleSize(k), 1)
	if err != nil || len(trained) != SampleSize(k) {
		t.Fatalf("sample = %d rows, %v; want SampleSize(%d) = %d", len(trained), err, k, SampleSize(k))
	}
	fit, _, _ := Run(context.Background(), trained, k, 10, 1)
	for i := range centroids {
		for d := range centroids[i] {
			if centroids[i][d] != fit[i][d] {
				t.Fatalf("RunSampled centroid %d dim %d = %v, fit over the sample = %v", i, d, centroids[i][d], fit[i][d])
			}
		}
	}
	if len(centroids) != k || len(assign) != n {
		t.Fatalf("centroids=%d assign=%d, want %d and %d", len(centroids), len(assign), k, n)
	}
	// Every vector assigned, and assigned to its true nearest — the sample
	// picks the frame, the full pass places the corpus.
	for i := range assign {
		if assign[i] < 0 || assign[i] >= k {
			t.Fatalf("assign[%d] = %d out of range", i, assign[i])
		}
		if i%997 == 0 {
			if want := Nearest(centroids, vecs[i]); assign[i] != want {
				t.Fatalf("assign[%d] = %d, want nearest %d", i, assign[i], want)
			}
		}
	}
	// A corpus the sample already covers gets the full fit, unchanged.
	small := vecs[:SampleSize(k)]
	got, _, _ := (Config{K: k, Iters: 10, Seed: 1}).Fit(context.Background(), small)
	want, _, _ := Run(context.Background(), small, k, 10, 1)
	for i := range got {
		for d := range got[i] {
			if got[i][d] != want[i][d] {
				t.Fatalf("RunSampled != Run when the sample covers the corpus (centroid %d, dim %d)", i, d)
			}
		}
	}
}

func TestSampleSizeForBudget(t *testing.T) {
	if got := SampleSizeForBudget(1000, 128, 1<<20); got > (1<<20)/(128*4) {
		t.Fatalf("sample has %d vectors, exceeds byte budget", got)
	}
	if got := SampleSizeForBudget(1000, 128, 1); got != 1 {
		t.Fatalf("minimum sample = %d, want 1", got)
	}
}

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
				Run(context.Background(), vecs, c.k, iters, 1)
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
				sink = Nearest(cents, q)
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
					sink = Nearest(cents, q)
				}
			})
			b.StopTimer()
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())*float64(runtime.GOMAXPROCS(0))/float64(b.N)/float64(k), "ns/centroid")
		})
	}
}

var sink int
