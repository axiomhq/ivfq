package kmeans

import (
	"context"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/axiomhq/ivfq/internal/simd"
)

// TestFitEMeansMatchesFit is the gate for choosing FitEMeans over Fit: on
// overlapping clusters, sampled, its squared error is within 5% of Fit's,
// under either batch law (above and at most 16 dimensions), and every vector
// sits in its nearest cluster. The random start alone measures 1.5 to 1.8
// times Fit's error here, and one pass 1.07 at 32 dimensions.
func TestFitEMeansMatchesFit(t *testing.T) {
	sse := func(cents, vecs [][]float32, assign []int) (sum float64) {
		for i, c := range assign {
			sum += float64(simd.L2Sq(cents[c], vecs[i]))
		}
		return sum
	}
	for _, c := range []struct {
		name       string
		n, dims, k int
		sigma      float64
	}{
		{"d32", 20_000, 32, 100, 1},
		{"d8", 50_000, 8, 256, 0.5},
	} {
		t.Run(c.name, func(t *testing.T) {
			vecs := clusters(41, c.n, c.dims, 100, c.sigma)
			fc, fa, err := Config{K: c.k, Iters: 11, Seed: 1}.Fit(ctx, vecs)
			if err != nil {
				t.Fatal(err)
			}
			ec, ea, err := Config{K: c.k, Iters: EMeansIters, Seed: 1}.FitEMeans(ctx, vecs)
			if err != nil {
				t.Fatal(err)
			}
			if len(ec) != c.k || len(ea) != c.n {
				t.Fatalf("shape: %d centroids, %d assigns", len(ec), len(ea))
			}
			for i := 0; i < c.n; i += 97 {
				if want := Nearest(ec, vecs[i]); ea[i] != want {
					t.Fatalf("assign[%d] = %d, want nearest %d", i, ea[i], want)
				}
			}
			if got, want := sse(ec, vecs, ea), sse(fc, vecs, fa); got > want*1.05 {
				t.Fatalf("FitEMeans SSE %.4g is more than 5%% above Fit's %.4g", got, want)
			}
		})
	}
}

// TestFitEMeansWorkerCountInvariant: the batch assignment fans out (1,024
// rows x 256 centroids is above parallelMin), and it must not change the
// answer.
func TestFitEMeansWorkerCountInvariant(t *testing.T) {
	vecs := clusters(9, 8192, 24, 64, 1)
	cfg := Config{K: 256, Iters: EMeansIters, Seed: 11}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	c1, a1, _ := cfg.FitEMeans(ctx, vecs)
	runtime.GOMAXPROCS(16)
	c2, a2, _ := cfg.FitEMeans(ctx, vecs)
	if len(c1) != 256 || !reflect.DeepEqual(c1, c2) || !reflect.DeepEqual(a1, a2) {
		t.Fatal("FitEMeans depends on GOMAXPROCS: 1 worker and 16 disagree")
	}
}

func TestFitEMeansSpherical(t *testing.T) {
	vecs := clusters(6, 5_000, 24, 20, 1)
	for _, v := range vecs {
		simd.Normalize(v)
	}
	cents, _, err := Config{K: 20, Iters: EMeansIters, Seed: 3, Spherical: true}.FitEMeans(ctx, vecs)
	if err != nil || len(cents) != 20 {
		t.Fatalf("FitEMeans = %d centroids, %v", len(cents), err)
	}
	for i, c := range cents {
		if n := simd.Dot(c, c); n < 0.999 || n > 1.001 {
			t.Fatalf("spherical centroid %d has |c|^2 %v", i, n)
		}
	}
}

func TestFitEMeansEdge(t *testing.T) {
	vecs := [][]float32{{1, 1}, {2, 2}}
	for k, want := range map[int]int{10: 2, 0: 1} { // K clamps to [1, n]
		for _, iters := range []int{0, EMeansIters} {
			if c, a, _ := (Config{K: k, Iters: iters, Seed: 1}).FitEMeans(ctx, vecs); len(c) != want || len(a) != 2 {
				t.Fatalf("K=%d Iters=%d: %d centroids, %d assigned", k, iters, len(c), len(a))
			}
		}
	}
	if c, a, err := (Config{K: 3, Iters: EMeansIters}).FitEMeans(ctx, nil); c != nil || a != nil || err != nil {
		t.Fatal("empty input should return nils")
	}
	// A MaxBytes sample of one row lowers K to it, as in Fit.
	if c, a, _ := (Config{K: 2, Iters: EMeansIters, MaxBytes: 8}).FitEMeans(ctx, vecs); len(c) != 1 || len(a) != 2 {
		t.Fatalf("MaxBytes: %d centroids, %d assigned", len(c), len(a))
	}
	// Above TwoLevelK it is Fit.
	big := clusters(3, 2_000, 4, 50, 0.3)
	cfg := Config{K: TwoLevelK + 1, Iters: 3, Seed: 5}
	fc, fa, _ := cfg.Fit(ctx, big)
	ec, ea, _ := cfg.FitEMeans(ctx, big)
	if len(ec) != TwoLevelK+1 || !reflect.DeepEqual(fc, ec) || !reflect.DeepEqual(fa, ea) {
		t.Fatal("FitEMeans above TwoLevelK is not Fit")
	}
}

// TestFitEMeansReseedsEmpty: a cluster the assignment leaves empty takes
// half of a split one, and every vector still sits in its nearest cluster.
// All three centroids start on copies of one row, the first to move takes
// the four far rows, and of the two left on the copies only the lower index
// wins them (two dimensions, so revival does not step in either). The far
// rows are nearer to their mean than to each other, so a centroid put on
// one of them would win it alone.
func TestFitEMeansReseedsEmpty(t *testing.T) {
	vecs := [][]float32{{8, 7.5}, {10, 7.5}, {8, 10.5}, {10, 10.5}}
	for range 300 {
		vecs = append(vecs, []float32{1, 1})
	}
	for seed := range int64(8) {
		cents, assign, err := (Config{K: 3, Iters: EMeansIters, Seed: seed}).FitEMeans(ctx, vecs)
		if err != nil {
			t.Fatal(err)
		}
		sizes := make([]int, 3)
		for i, c := range assign {
			sizes[c]++
			if c != Nearest(cents, vecs[i]) {
				t.Fatalf("seed %d: vector %d is not in its nearest cluster", seed, i)
			}
		}
		if slices.Min(sizes) < 2 {
			t.Errorf("seed %d: cluster sizes %v", seed, sizes)
		}
	}
}

// TestFitEMeansCancelled: a cancelled context returns its error and no
// result, whether it was cancelled before the fit or inside training.
func TestFitEMeansCancelled(t *testing.T) {
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	vecs, k := sampledCorpus()
	cfg := Config{K: k, Iters: EMeansIters, Seed: 1}
	for name, c := range map[string]context.Context{
		"before, sampled": cancelled,
		"in training":     &cancelAfterPolls{Context: ctx, at: 20}, // sampling takes 4 polls
	} {
		if c, a, err := cfg.FitEMeans(c, vecs); err != context.Canceled || c != nil || a != nil {
			t.Fatalf("%s: FitEMeans = (%v, %d assigned, %v)", name, c, len(a), err)
		}
	}
	if c, a, err := cfg.FitEMeans(cancelled, vecs[:100]); err != context.Canceled || c != nil || a != nil {
		t.Fatalf("before, whole corpus: FitEMeans = (%v, %d assigned, %v)", c, len(a), err)
	}
}

// BenchmarkFitEMeans is Fit against FitEMeans end to end at the largest
// flat fit a HoodK build asks for, 100k x 128 at K = 511:
//
//	go test -run x -bench FitEMeans -benchtime 1x -count 10 ./kmeans/
func BenchmarkFitEMeans(b *testing.B) {
	vecs := benchCorpus(100_000, 128, 31)
	for _, c := range []struct {
		name  string
		fit   func(Config, context.Context, [][]float32) ([][]float32, []int, error)
		iters int
	}{{"fit", Config.Fit, 11}, {"emeans", Config.FitEMeans, EMeansIters}} {
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				if _, _, err := c.fit(Config{K: 511, Iters: c.iters, Seed: 1}, ctx, vecs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
