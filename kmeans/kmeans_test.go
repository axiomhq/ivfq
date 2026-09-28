package kmeans_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/axiomhq/ivfq/kmeans"
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
	centroids, assign, _ := kmeans.Run(ctx, vecs, 3, 10, 42)
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
		if kmeans.Nearest(centroids, v) != assign[i] {
			t.Fatalf("assign[%d] disagrees with Nearest", i)
		}
	}
}

func TestDeterminism(t *testing.T) {
	vecs := blobs(2, 30, []float32{0, 0}, []float32{5, 5})
	c1, a1, _ := kmeans.Run(ctx, vecs, 2, 10, 7)
	c2, a2, _ := kmeans.Run(ctx, vecs, 2, 10, 7)
	if !reflect.DeepEqual(c1, c2) || !reflect.DeepEqual(a1, a2) {
		t.Fatal("same seed, different output")
	}
}

func TestClampAndEdge(t *testing.T) {
	vecs := [][]float32{{1, 1}, {2, 2}}
	c, a, _ := kmeans.Run(ctx, vecs, 10, 5, 1) // k > n clamps to n
	if len(c) != 2 || len(a) != 2 {
		t.Fatalf("clamp: %d centroids", len(c))
	}
	if c2, a2, _ := kmeans.Run(ctx, nil, 3, 5, 1); c2 != nil || a2 != nil {
		t.Fatal("empty input should return nils")
	}
}

func TestReseedDistinct(t *testing.T) {
	// Duplicate-heavy input with k above the natural cluster count forces
	// simultaneous empty clusters: each must reseed to a DISTINCT vector
	// (review fix — un-claimed reseeds all cloned one vector and the
	// duplicates were a stable fixed point).
	vecs := [][]float32{{0, 0}, {0, 0}, {100, 100}, {100, 100}, {50, 0}, {0, 50}}
	centroids, _, _ := kmeans.Run(ctx, vecs, 4, 8, 0)
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
		centroids, _, _ := kmeans.Run(ctx, vecs, 5, 3, seed)
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
	centroids, _, _ := kmeans.RunSpherical(ctx, vecs, 2, 10, 3)
	for i, c := range centroids {
		var n float64
		for _, x := range c {
			n += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(n)-1) > 1e-3 {
			t.Fatalf("centroid %d not unit: |c|=%f", i, math.Sqrt(n))
		}
	}
	c2, _, _ := kmeans.RunSpherical(ctx, vecs, 2, 10, 3)
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
	c1, a1, _ := kmeans.Run(ctx, vecs, 64, 10, 11)  // n*k = 128k, above ivf's parallel floor
	runtime.GOMAXPROCS(16)
	c2, a2, _ := kmeans.Run(ctx, vecs, 64, 10, 11)
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
	if c, a, err := kmeans.Run(cancelled, vecs, 2, 10, 1); err != context.Canceled || c != nil || a != nil {
		t.Fatalf("Run on cancelled ctx = (%v, %v, %v)", c, a, err)
	}
	big := blobs(5, kmeans.SampleSize(2)+1, []float32{0, 0}, []float32{5, 5})
	if c, a, err := (kmeans.Config{K: 2, Iters: 10, Seed: 1}).Fit(cancelled, big); err != context.Canceled || c != nil || a != nil {
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
	if _, assign, err := (kmeans.Config{K: k, Iters: 2, Seed: 1}).Fit(ctx, vecs); err != nil || len(assign) != n {
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
	n := 4 * kmeans.SampleSize(k)
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
	c, a, err := (kmeans.Config{K: k, Iters: 10, Seed: 1}).Fit(pc, vecs)
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
	if _, _, err := (kmeans.Config{K: k, Iters: 10, Seed: 1}).Fit(full, vecs); err != nil {
		t.Fatal(err)
	}
	chunks := int64((len(vecs) + 255) / 256)
	at := full.polls.Load() - chunks/2
	pc := &cancelAfterPolls{Context: ctx, at: at}
	c, a, err := (kmeans.Config{K: k, Iters: 10, Seed: 1}).Fit(pc, vecs)
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
