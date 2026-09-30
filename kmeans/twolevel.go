package kmeans

import (
	"context"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/axiomhq/ivfq/internal/simd"
)

// TwoLevelK is the K above which Fit trains two-level, so that seeding and
// every Lloyd pass cost O(n·√K·d) instead of O(n·K·d):
//
//  1. ceil(√K) coarse centroids by the flat fit on the sample;
//  2. a flat fit per coarse bucket for its share of K, proportional to the
//     bucket's size and at least one, concatenated;
//  3. min(Iters, twoLevelRefine) global Lloyd iterations over those K, and
//     the final assignment, each searching only the fine centroids filed
//     under a vector's twoLevelProbe nearest coarse centroids.
//
// Step 3 is what makes it match the flat fit: without it a true cluster cut
// by a bucket boundary stays cut, measured +22% SSE and -1.8 points of
// recall@10 at K = 2,000. Callers need no change; below the threshold the
// flat fit is cheap enough.
const TwoLevelK = 512

// twoLevelProbe and twoLevelRefine were picked on the adversarial case, 2,000
// iid gaussian clusters in 128 dims at K = 2,000, where the coarse level
// carries little structure (SSE against the flat fit, under Fit's assign):
//
//	probe 4, refine 10:  +15.7%
//	probe 8, refine 10:   +7.4%
//	probe 12, refine 3:   +4.8%
//	probe 16, refine 3:   +2.9%
//	probe 16, refine 1:   +7.6%
//
// A vector then costs (1+probe)·√K distance calls a pass against K flat.
const (
	twoLevelProbe  = 16
	twoLevelRefine = 3
)

// twoLevel is a two-level fit: owned[b] are the fine centroids nearest
// coarse centroid b.
type twoLevel struct {
	coarse [][]float32
	fine   [][]float32
	owned  [][]int
}

// fitTwoLevel is Fit above TwoLevelK: train on a sample of s vectors (all
// of them when s >= len(vecs)), then assign every vector.
func fitTwoLevel(ctx context.Context, vecs [][]float32, s, k, iters int, seed int64, spherical bool) ([][]float32, []int, error) {
	if s >= len(vecs) {
		t, assign, err := trainTwoLevel(ctx, vecs, k, iters, seed, spherical)
		if err != nil {
			return nil, nil, err
		}
		return t.fine, assign, nil
	}
	train, err := sample(ctx, vecs, s, seed)
	if err != nil {
		return nil, nil, err
	}
	t, _, err := trainTwoLevel(ctx, train, min(k, len(train)), iters, seed, spherical)
	if err != nil {
		return nil, nil, err
	}
	assign := make([]int, len(vecs))
	for i := range assign {
		assign[i] = -1
	}
	t.assign(ctx, vecs, assign)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return t.fine, assign, nil
}

func trainTwoLevel(ctx context.Context, vecs [][]float32, k, iters int, seed int64, spherical bool) (*twoLevel, []int, error) {
	kc := int(math.Ceil(math.Sqrt(float64(k))))
	coarse, top, err := run(ctx, vecs, kc, iters, seed, spherical)
	if err != nil {
		return nil, nil, err
	}
	all := make([][]int, len(coarse))
	for i, c := range top {
		all[c] = append(all[c], i)
	}
	// An empty bucket gets no fine centroids: drop it.
	t := &twoLevel{}
	var buckets [][]int
	var sizes []int
	for c, b := range all {
		if len(b) > 0 {
			t.coarse = append(t.coarse, coarse[c])
			buckets = append(buckets, b)
			sizes = append(sizes, len(b))
		}
	}
	quota := shares(sizes, k)
	first := make([]int, len(buckets)+1)
	for b, q := range quota {
		first[b+1] = first[b] + q
	}
	t.fine = make([][]float32, k)
	assign := make([]int, len(vecs))
	// Buckets are independent fits, each deterministic for its seed, so
	// running them concurrently keeps the result independent of the worker
	// count; each fit also fans out internally when its bucket is large.
	// One at a time left a small bucket's seeding serial.
	err = parallel(len(buckets), func(b int) error {
		vs := make([][]float32, len(buckets[b]))
		for j, i := range buckets[b] {
			vs[j] = vecs[i]
		}
		fine, local, err := run(ctx, vs, quota[b], iters, seed+int64(b)+1, spherical)
		if err != nil {
			return err
		}
		copy(t.fine[first[b]:first[b+1]], fine)
		for j, i := range buckets[b] {
			assign[i] = first[b] + local[j]
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	_, _, err = lloyd(ctx, vecs, t.fine, assign, min(iters, twoLevelRefine), spherical, func(assign []int) bool {
		return t.assign(ctx, vecs, assign)
	})
	if err != nil {
		return nil, nil, err
	}
	return t, assign, nil
}

// assign files every fine centroid under its nearest coarse centroid and
// sets each assign[i] to vecs[i]'s nearest fine centroid among its
// twoLevelProbe nearest buckets.
func (t *twoLevel) assign(ctx context.Context, vecs [][]float32, assign []int) bool {
	t.owned = make([][]int, len(t.coarse))
	for f, c := range t.fine {
		b := Nearest(t.coarse, c)
		t.owned[b] = append(t.owned[b], f)
	}
	probe := min(twoLevelProbe, len(t.coarse))
	work := len(t.coarse) + probe*len(t.fine)/len(t.coarse)
	return assignWith(ctx, len(vecs), assign, work, func(i int) int {
		top, m := nearestProbe(t.coarse, vecs[i])
		best, bestD := 0, float32(math.MaxFloat32)
		for _, b := range top[:m] {
			for _, f := range t.owned[b] {
				if d := simd.L2Sq(t.fine[f], vecs[i]); d < bestD || (d == bestD && f < best) {
					best, bestD = f, d
				}
			}
		}
		return best
	})
}

// nearestProbe is the indices of the (up to) twoLevelProbe nearest
// centroids to v, nearest first, in an array so the per-vector call does
// not allocate.
func nearestProbe(centroids [][]float32, v []float32) ([twoLevelProbe]int, int) {
	var top [twoLevelProbe]int
	var topD [twoLevelProbe]float32
	m := 0
	for i, c := range centroids {
		d := simd.L2Sq(c, v)
		if m == len(top) && d >= topD[m-1] {
			continue
		}
		j := min(m, len(top)-1)
		for j > 0 && topD[j-1] > d {
			top[j], topD[j] = top[j-1], topD[j-1]
			j--
		}
		top[j], topD[j] = i, d
		m = min(m+1, len(top))
	}
	return top, m
}

// shares splits k fine centroids over buckets of the given sizes (all > 0,
// summing to at least k): one each, then proportionally by the D'Hondt
// rule, never more centroids than a bucket has vectors.
func shares(sizes []int, k int) []int {
	q := make([]int, len(sizes))
	for b := range q {
		q[b] = 1
	}
	// ponytail: O(k·buckets) = O(k^1.5) scan, 8e6 steps at K = 40,000; a
	// heap if K reaches the millions.
	for left := k - len(sizes); left > 0; left-- {
		best := -1
		for b, s := range sizes {
			if q[b] < s && (best < 0 || s*(q[best]+1) > sizes[best]*(q[b]+1)) {
				best = b
			}
		}
		q[best]++
	}
	return q
}

// parallel runs f(0..n-1) over GOMAXPROCS workers and returns the first
// error by index. Each f writes only its own slots, so results do not
// depend on the worker count.
func parallel(n int, f func(int) error) error {
	var next atomic.Int64
	var wg sync.WaitGroup
	errs := make([]error, n)
	for range min(runtime.GOMAXPROCS(0), n) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				errs[i] = f(i)
			}
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
