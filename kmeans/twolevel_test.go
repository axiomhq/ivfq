package kmeans

import (
	"context"
	"math/rand"
	"reflect"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/axiomhq/ivfq/internal/simd"
)

func TestShares(t *testing.T) {
	sizes := []int{1, 3, 1000, 50, 200}
	q := shares(sizes, 100)
	sum := 0
	for b, n := range q {
		if n < 1 || n > sizes[b] {
			t.Fatalf("bucket %d (size %d) got %d", b, sizes[b], n)
		}
		sum += n
	}
	if sum != 100 {
		t.Fatalf("sum %d, want 100", sum)
	}
	// Proportional where the caps do not bind: 1000 of 1254 vectors.
	if q[2] < 75 || q[2] > 82 {
		t.Fatalf("largest bucket got %d of 100: %v", q[2], q)
	}
	// k == total: every vector its own centroid.
	if got := shares([]int{2, 5, 3}, 10); !reflect.DeepEqual(got, []int{2, 5, 3}) {
		t.Fatalf("exact split %v", got)
	}
}

// clusters is n vectors of dims around k gaussian centres of unit spread per
// dimension, noise sigma — overlapping enough that the fit matters.
func clusters(seed int64, n, dims, k int, sigma float64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, k)
	for i := range centers {
		centers[i] = make([]float32, dims)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(k)]
		v := make([]float32, dims)
		for d := range v {
			v[d] = c[d] + float32(rng.NormFloat64()*sigma)
		}
		out[i] = v
	}
	return out
}

func TestTwoLevelFit(t *testing.T) {
	vecs := clusters(3, 20_000, 16, 400, 0.3)
	const k = 700
	c1, a1, err := Config{K: k, Iters: 6, Seed: 5}.Fit(ctx, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if len(c1) != k || len(a1) != len(vecs) {
		t.Fatalf("shape: %d centroids, %d assigns", len(c1), len(a1))
	}
	members := make([]int, k)
	for _, c := range a1 {
		members[c]++
	}
	empty := 0
	for _, m := range members {
		if m == 0 {
			empty++
		}
	}
	if empty > k/100 {
		t.Fatalf("%d of %d centroids empty", empty, k)
	}
	assertDistinct(t, c1)

	prev := runtime.GOMAXPROCS(1)
	c2, a2, err := Config{K: k, Iters: 6, Seed: 5}.Fit(ctx, vecs)
	runtime.GOMAXPROCS(prev)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c1, c2) || !reflect.DeepEqual(a1, a2) {
		t.Fatal("two-level fit depends on GOMAXPROCS")
	}

	sc, _, err := Config{K: k, Iters: 6, Seed: 5, Spherical: true}.Fit(ctx, vecs)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range sc {
		if n := simd.Dot(c, c); n < 0.999 || n > 1.001 {
			t.Fatalf("spherical centroid %d has |c|^2 %v", i, n)
		}
	}

	// Sampled: the budget caps the training set, every vector is assigned.
	sampled, assign, err := Config{K: k, Iters: 6, Seed: 5, MaxBytes: 16 * 4 * 5_000}.Fit(ctx, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if len(sampled) != k || len(assign) != len(vecs) {
		t.Fatalf("sampled shape: %d centroids, %d assigns", len(sampled), len(assign))
	}

	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if c, a, err := (Config{K: k, Iters: 6}).Fit(cctx, vecs); err == nil || c != nil || a != nil {
		t.Fatalf("cancelled fit returned %d centroids, err %v", len(c), err)
	}
}

// TestTwoLevelQuality is the gate for the two-level fit: at K = 2,000 on
// 200k x 128 with 2,000 true clusters, SSE within 5% of the flat fit and
// recall@10 of an IVF probe within one point. nprobe = 98 is
// ivfq.DefaultNprobe(2000) (the package cannot import ivfq).
func TestTwoLevelQuality(t *testing.T) {
	if testing.Short() {
		t.Skip("200k-vector fits")
	}
	const n, dims, k, nq, nprobe, topK = 200_000, 128, 2_000, 500, 98, 10
	all := clusters(11, n+nq, dims, k, 1)
	vecs, queries := all[:n], all[n:]
	truth := make([][]int, nq)
	parallelFor(nq, func(q int) { truth[q] = bruteTop(vecs, nil, queries[q], topK) })

	type result struct {
		sse, recall float64
		wall        time.Duration
		empty       int
	}
	eval := func(fit func() ([][]float32, []int, error)) result {
		start := time.Now()
		cents, assign, err := fit()
		wall := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		var r result
		r.wall = wall
		lists := make([][]int, len(cents))
		for i, c := range assign {
			lists[c] = append(lists[c], i)
			r.sse += float64(simd.L2Sq(cents[c], vecs[i]))
		}
		for _, l := range lists {
			if len(l) == 0 {
				r.empty++
			}
		}
		hits := make([]int, nq)
		parallelFor(nq, func(q int) {
			probe := bruteTop(cents, nil, queries[q], nprobe)
			var cand []int
			for _, c := range probe {
				cand = append(cand, lists[c]...)
			}
			got := bruteTop(vecs, cand, queries[q], topK)
			want := map[int]bool{}
			for _, id := range truth[q] {
				want[id] = true
			}
			for _, id := range got {
				if want[id] {
					hits[q]++
				}
			}
		})
		total := 0
		for _, h := range hits {
			total += h
		}
		r.recall = float64(total) / float64(nq*topK)
		return r
	}
	s := SampleSize(k)
	flat := eval(func() ([][]float32, []int, error) { return fitFlat(ctx, vecs, s, k, 10, 1, false) })
	two := eval(func() ([][]float32, []int, error) { return fitTwoLevel(ctx, vecs, s, k, 10, 1, false) })
	t.Logf("flat:      SSE %.4g  recall@10 %.4f  empty %d  wall %v", flat.sse, flat.recall, flat.empty, flat.wall)
	t.Logf("two-level: SSE %.4g  recall@10 %.4f  empty %d  wall %v", two.sse, two.recall, two.empty, two.wall)
	if two.sse > flat.sse*1.05 {
		t.Errorf("two-level SSE %.4g is more than 5%% above flat %.4g", two.sse, flat.sse)
	}
	if two.recall < flat.recall-0.01 {
		t.Errorf("two-level recall %.4f is more than a point below flat %.4f", two.recall, flat.recall)
	}
}

// BenchmarkFitLargeK is the wall-time gate: K = 20,000 on 1M x 128, flat
// against two-level, 10 iterations as a build runs them.
//
//	go test -run x -bench FitLargeK -benchtime 1x ./kmeans/
func BenchmarkFitLargeK(b *testing.B) {
	const n, dims, k, iters = 1_000_000, 128, 20_000, 10
	vecs := clusters(13, n, dims, k, 1)
	s := SampleSize(k)
	for _, c := range []struct {
		name string
		fit  func(context.Context, [][]float32, int, int, int, int64, bool) ([][]float32, []int, error)
	}{{"two-level", fitTwoLevel}, {"flat", fitFlat}} {
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				if _, _, err := c.fit(ctx, vecs, s, k, iters, 1, false); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// bruteTop is the k nearest of vecs to q (among ids, or all when nil).
func bruteTop(vecs [][]float32, ids []int, q []float32, k int) []int {
	if ids == nil {
		ids = make([]int, len(vecs))
		for i := range ids {
			ids[i] = i
		}
	}
	type hit struct {
		id int
		d  float32
	}
	hits := make([]hit, len(ids))
	for i, id := range ids {
		hits[i] = hit{id, simd.L2Sq(vecs[id], q)}
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].d < hits[j].d || hits[i].d == hits[j].d && hits[i].id < hits[j].id })
	out := make([]int, 0, k)
	for _, h := range hits[:min(k, len(hits))] {
		out = append(out, h.id)
	}
	return out
}

func parallelFor(n int, f func(int)) {
	var wg sync.WaitGroup
	next := make(chan int)
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				f(i)
			}
		}()
	}
	for i := range n {
		next <- i
	}
	close(next)
	wg.Wait()
}
