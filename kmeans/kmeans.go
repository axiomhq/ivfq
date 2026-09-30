package kmeans

import (
	"context"
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
)

// trainFactor sets the k-means training sample at trainFactor x k vectors.
// Lloyd's needs enough points per centroid to place it, not all of them: at
// 32 the sample is 3% of a 1M corpus and moves recall by less than the
// run-to-run noise. Below
// ~16 the fit degrades; above ~64 nothing improves and the training term
// starts to matter again.
const trainFactor = 32

// minSample floors the training set in absolute terms. The factor alone is
// fine as a RATIO but not as a COUNT: at k=20 it asks for 640 points, and a
// 640-point fit is at the mercy of one random init — measured 7.74x the full
// fit's MSE on an 8-blob corpus where 320 and 1,280 points both landed within
// 1.05x. 4,096 costs nothing (the training term is a rounding error against
// the one O(N x k) assignment pass at any N where sampling engages at all)
// and holds every measured case inside 1.03x of the full fit.
const minSample = 4096

// SampleSize is the training-set bound: NOTHING in a build may call Lloyd's
// with more vectors than this. It is the property that makes the build
// linear-ish in N — training is O(SampleSize(k) x k x dims) whatever N is,
// and the only pass that touches all N vectors is one assignment.
func SampleSize(k int) int { return max(minSample, trainFactor*max(1, k)) }

// SampleSizeForBudget additionally caps the training vectors by their raw
// float32 footprint. It never returns zero for a non-empty build.
func SampleSizeForBudget(k, dims, bytes int) int {
	if dims <= 0 || bytes <= 0 {
		return SampleSize(k)
	}
	return max(1, min(SampleSize(k), bytes/(dims*4)))
}

// Config is one k-means fit.
type Config struct {
	// K is how many centroids to fit, clamped to [1, len(vecs)].
	K int
	// Iters is how many Lloyd iterations follow the k-means++ seeding.
	Iters int
	// Seed fixes the training sample, the seeding draw and every reseed,
	// so one Config over one input gives one answer.
	Seed int64
	// Spherical renormalizes every centroid to unit length after each
	// mean, for cosine fields: plain L2 against unit centroids is then
	// monotone in cosine, so assignment stays L2 everywhere.
	Spherical bool
	// MaxBytes caps the training sample by its float32 footprint on top of
	// SampleSize(K). Zero means SampleSize(K) alone.
	MaxBytes int
}

// SplitIters is the iteration count a cluster split uses:
// Config{K: 2, Iters: SplitIters}.
const SplitIters = 8

// Fit fits c.K centroids on a deterministic random sample of at most
// SampleSize(c.K) vectors, then assigns EVERY vector in one parallel pass.
// It returns the centroids and each vector's centroid index. This is the
// only k-means a build runs at scale: with fixed-size clusters k = N/1024,
// so Lloyd's over all N would be O(N^2/S) per iteration. Sampling collapses
// the iterated term to the sample and leaves exactly one O(N x k) pass.
//
// When the sample covers the corpus the result is the plain fit over every
// vector. An empty corpus returns nil, nil, nil. A cancelled ctx returns
// its error and no result: checked at every seeding step, every Lloyd
// iteration, and every assignment chunk. The result is bit-identical
// whatever GOMAXPROCS is.
//
// Above TwoLevelK the fit is two-level (see TwoLevelK): same inputs, same
// guarantees, except that assign is the nearest centroid within the
// vector's nearest few coarse buckets rather than over all K.
func (c Config) Fit(ctx context.Context, vecs [][]float32) ([][]float32, []int, error) {
	k, iters, seed, spherical, bytes := c.K, c.Iters, c.Seed, c.Spherical, c.MaxBytes
	n := len(vecs)
	if n == 0 {
		return nil, nil, nil
	}
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	s := SampleSizeForBudget(k, len(vecs[0]), bytes)
	if k > TwoLevelK {
		return fitTwoLevel(ctx, vecs, s, k, iters, seed, spherical)
	}
	return fitFlat(ctx, vecs, s, k, iters, seed, spherical)
}

// fitFlat is Fit at or below TwoLevelK: Lloyd's over all k on the sample.
func fitFlat(ctx context.Context, vecs [][]float32, s, k, iters int, seed int64, spherical bool) ([][]float32, []int, error) {
	n := len(vecs)
	if s >= n {
		return run(ctx, vecs, k, iters, seed, spherical)
	}
	sample, err := sample(ctx, vecs, s, seed)
	if err != nil {
		return nil, nil, err
	}
	centroids, _, err := run(ctx, sample, k, iters, seed, spherical)
	if err != nil {
		return nil, nil, err
	}
	assign := make([]int, n)
	assignAll(ctx, centroids, vecs, assign)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return centroids, assign, nil
}

// sample draws s of vecs uniformly at random, in input order. It is the
// only rows Fit ever hands Lloyd's, which is what bounds training
// at SampleSize(k) whatever len(vecs) is.
//
// A uniform random sample, not a stride: the corpus arrives sorted by id
// and generators commonly make attribute i a function of i mod something,
// so a stride can land every sampled vector in the same mode. Reservoir
// sampling (Algorithm R) keeps the scratch at s indices whatever n is;
// the chosen indices are then sorted so the training order is the input
// order, independent of the draw sequence. This pass touches all n rows,
// so ctx is polled between fixed-size chunks of it, as assignAll does.
func sample(ctx context.Context, vecs [][]float32, s int, seed int64) ([][]float32, error) {
	const reservoirChunk = 4096
	n := len(vecs)
	rng := rand.New(rand.NewSource(seed))
	idx := make([]int, s)
	for i := range idx {
		idx[i] = i
	}
	for lo := s; lo < n; lo += reservoirChunk {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i := lo; i < min(lo+reservoirChunk, n); i++ {
			if j := rng.Intn(i + 1); j < s {
				idx[j] = i
			}
		}
	}
	slices.Sort(idx)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, s)
	for i, j := range idx {
		out[i] = vecs[j]
	}
	return out, nil
}

// seedPlusPlus picks the k initial centroids by D^2 sampling — k-means++.
// The first is uniform; every one after it is drawn with probability
// proportional to its squared distance from the nearest centroid already
// chosen, so the seeds spread over the corpus instead of clumping wherever
// the density happens to be.
//
// With k = N/1024 a build asks for thousands of centroids, and a uniform
// init strands a large share of them in near-empty space next to a few
// enormous clusters: at 10M it left 74% of clusters outside [S/4, 2S] with
// a p25 of four vectors.
//
// Cost is k passes, each one distance per vector: k*n L2Sq calls, against the
// (iters+1)*n*k Lloyd's does — about one extra iteration's worth, and it
// usually pays for itself by converging in fewer. The per-step update is
// parallel and each index is written independently, so like assignAll the
// result is bit-identical at any worker count; the draws come from rng alone,
// so it is deterministic for a given seed.
func seedPlusPlus(ctx context.Context, vecs [][]float32, k int, rng *rand.Rand) ([][]float32, error) {
	n := len(vecs)
	centroids := make([][]float32, 0, k)
	centroids = append(centroids, slices.Clone(vecs[rng.Intn(n)]))
	if k == 1 {
		return centroids, nil
	}
	// d2[i] is the squared distance from vecs[i] to its nearest chosen
	// centroid, carried forward so each step costs one distance, not len(c).
	d2 := make([]float32, n)
	for i := range d2 {
		d2[i] = math.MaxFloat32
	}
	for len(centroids) < k {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		minInto(d2, vecs, centroids[len(centroids)-1])
		var sum float64
		for _, d := range d2 {
			sum += float64(d)
		}
		if sum <= 0 {
			// Every remaining point sits on a centroid we already have: a
			// corpus with fewer than k distinct values. D^2 has nothing left
			// to say, so fall back to uniform and let the empty-cluster
			// reseed in run() sort out the duplicates.
			centroids = append(centroids, slices.Clone(vecs[rng.Intn(n)]))
			continue
		}
		// GREEDY k-means++: draw seedTrials candidates by D^2 and keep the one
		// that actually lowers the total cost most, rather than trusting a
		// single draw. Plain D^2 sampling picks a point in proportion to its
		// squared distance, which in high dimensions is a bias toward
		// OUTLIERS — a seed on the shell of a gaussian captures almost
		// nothing and leaves a tiny hood behind. Measured at 1M x 128 dims:
		// plain D^2 put a quarter of the hoods at 11 vectors or fewer.
		best, bestCost := -1, math.Inf(1)
		for try := 0; try < seedTrials; try++ {
			cand := drawD2(d2, sum, rng.Float64())
			if c := costWith(d2, vecs, vecs[cand]); c < bestCost {
				best, bestCost = cand, c
			}
		}
		centroids = append(centroids, slices.Clone(vecs[best]))
	}
	return centroids, nil
}

// seedTrials is the greedy candidate count. Standard practice is
// 2 + log(k); a flat 4 is within noise of that over the k range a 1M to
// 100M index asks for (977 to 97,657, so 2+log k is 9 to 13) and costs a quarter as
// much as 13 would. Cost is seedTrials * k * sample distance calls.
const seedTrials = 4

// drawD2 returns the index D^2 sampling picks for a uniform draw u in [0,1).
func drawD2(d2 []float32, sum float64, u float64) int {
	target, acc := u*sum, 0.0
	for i, d := range d2 {
		acc += float64(d)
		if acc >= target {
			return i
		}
	}
	return len(d2) - 1
}

// costWith is the total squared error if c joined the centroid set: the
// quantity greedy k-means++ minimizes over its candidates, and the seeding's
// dominant term at seedTrials * k * n distance calls (1.2e10 at 10M), so it
// fans out.
//
// It is deliberately EXACT over the whole training sample. Estimating it on a
// stride was measured and dropped: the whole benefit of greedy over plain D^2
// is in the ranking, and the ranking does not survive approximation. On the
// 1M x 128 corpus the share of hoods inside [S/4, 2S] went 60.2% at full cost
// to 44.2% / 39.4% / 44.5% at strides of 2 / 4 / 8 — back to what a uniform
// init gives, for a third off the seeding. Halving seedTrials instead cost
// almost as much (48.4%). Full cost or do not bother.
//
// Determinism, which a parallel float64 SUM does not get for free: each
// fixed-size chunk sums into its own slot and the slots are then added in
// index order, so the result is bit-identical at any worker count. That is
// the same guarantee assignAll gives and it is why the reduction is not just
// an atomic add.
func costWith(d2 []float32, vecs [][]float32, c []float32) float64 {
	n := len(vecs)
	chunkSum := func(lo, hi int) float64 {
		var sum float64
		for i := lo; i < hi; i++ {
			if d := simd.L2Sq(c, vecs[i]); d < d2[i] {
				sum += float64(d)
			} else {
				sum += float64(d2[i])
			}
		}
		return sum
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers < 2 || n < parallelMin {
		return chunkSum(0, n)
	}
	const chunk = 4096
	parts := make([]float64, (n+chunk-1)/chunk)
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				j := int(next.Add(1)) - 1
				if j >= len(parts) {
					return
				}
				lo := j * chunk
				parts[j] = chunkSum(lo, min(lo+chunk, n))
			}
		}()
	}
	wg.Wait()
	var sum float64
	for _, p := range parts {
		sum += p
	}
	return sum
}

// minInto lowers each d2[i] to the distance from vecs[i] to c when that is
// closer. Same fan-out and the same independence-per-index guarantee as
// assignAll, so the result does not depend on the worker count.
func minInto(d2 []float32, vecs [][]float32, c []float32) {
	n := len(vecs)
	step := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			if d := simd.L2Sq(c, vecs[i]); d < d2[i] {
				d2[i] = d
			}
		}
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers < 2 || n < parallelMin {
		step(0, n)
		return
	}
	const chunk = 256
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				lo := int(next.Add(chunk)) - chunk
				if lo >= n {
					return
				}
				step(lo, min(lo+chunk, n))
			}
		}()
	}
	wg.Wait()
}

func run(ctx context.Context, vecs [][]float32, k, iters int, seed int64, spherical bool) ([][]float32, []int, error) {
	n := len(vecs)
	if n == 0 {
		return nil, nil, nil
	}
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	rng := rand.New(rand.NewSource(seed))
	centroids, err := seedPlusPlus(ctx, vecs, k, rng)
	if err != nil {
		return nil, nil, err
	}
	if spherical {
		for _, c := range centroids {
			simd.Normalize(c)
		}
	}
	return lloyd(ctx, vecs, centroids, make([]int, n), iters, spherical, func(assign []int) bool {
		return assignAll(ctx, centroids, vecs, assign)
	})
}

// lloyd runs up to iters Lloyd iterations from centroids (updated in place)
// and assign (the assignment step's starting point, updated in place) with
// assignFn as the assignment step, then one last assignment.
func lloyd(ctx context.Context, vecs, centroids [][]float32, assign []int, iters int, spherical bool, assignFn func([]int) bool) ([][]float32, []int, error) {
	k := len(centroids)
	dims := len(vecs[0])
	sums := make([]float64, k*dims) // row c is centroid c's accumulator
	counts := make([]int, k)
	for it := range iters {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		changed := assignFn(assign)
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		clear(sums)
		clear(counts)
		for i, v := range vecs {
			c := assign[i]
			counts[c]++
			row := sums[c*dims : (c+1)*dims]
			for d, x := range v {
				row[d] += float64(x)
			}
		}
		claimed := map[int]bool{} // vectors already used to reseed THIS pass
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Reseed from the farthest UNCLAIMED vector. The claimed
				// set gives each simultaneous empty cluster a distinct
				// seed WITHOUT mutating assign — mutating assign could
				// steal a live cluster's sole member and mint a duplicate
				// at low iteration budgets (review fix, round 2).
				fi := farthest(vecs, centroids, assign, claimed)
				centroids[c] = slices.Clone(vecs[fi])
				if spherical {
					simd.Normalize(centroids[c])
				}
				claimed[fi] = true
				continue
			}
			row := sums[c*dims : (c+1)*dims]
			for d := range centroids[c] {
				centroids[c][d] = float32(row[d] / float64(counts[c]))
			}
			if spherical {
				simd.Normalize(centroids[c])
			}
		}
		if !changed && it > 0 {
			break
		}
	}
	assignFn(assign)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return centroids, assign, nil
}

// parallelMin is the assignment work (n*k distance calls) below which the
// serial loop wins: goroutine start-up costs more than it saves. 1<<16 is
// ~64k L2Sq calls, a few milliseconds of work.
const parallelMin = 1 << 16

// assignAll sets assign[i] to the nearest centroid of vecs[i] and reports
// whether any entry changed, fanning the pass out over GOMAXPROCS workers
// pulling fixed-size contiguous chunks. A cancelled ctx stops the pass,
// serial or parallel, at the next chunk; the caller checks ctx.Err() and
// discards the pass.
//
// Determinism: every index is computed independently and written to its own
// slot, so the output is BIT-IDENTICAL to the serial loop for any worker
// count, any chunk schedule, and any GOMAXPROCS — that is the guarantee, and
// it is why the per-iteration centroid means below stay serial (they are
// O(n*dims) against this pass's O(n*k*dims), so parallelizing the reduction
// would buy ~0.1% at k=1000 while putting float64 summation order at the
// mercy of the worker count). NOT guaranteed: identical results across
// architectures, Go versions, or any change to L2Sq's summation
// order — floating-point k-means is only ever reproducible against a fixed
// kernel.
func assignAll(ctx context.Context, centroids, vecs [][]float32, assign []int) bool {
	return assignWith(ctx, len(vecs), assign, len(centroids), func(i int) int { return Nearest(centroids, vecs[i]) })
}

// assignWith is assignAll for any nearest-centroid rule: assign[i] =
// nearest(i) for i < n, which may read assign[i] before it is replaced.
// work is the distance calls one vector costs, for the serial cutoff.
func assignWith(ctx context.Context, n int, assign []int, work int, nearest func(int) int) bool {
	const chunk = 256
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers < 2 || n*work < parallelMin {
		changed := false
		for lo := 0; lo < n && ctx.Err() == nil; lo += chunk {
			if assignRange(nearest, assign, lo, min(lo+chunk, n)) {
				changed = true
			}
		}
		return changed
	}
	var next atomic.Int64
	var changed atomic.Bool
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				lo := int(next.Add(chunk)) - chunk
				if lo >= n {
					return
				}
				if assignRange(nearest, assign, lo, min(lo+chunk, n)) {
					changed.Store(true)
				}
			}
		}()
	}
	wg.Wait()
	return changed.Load()
}

func assignRange(nearest func(int) int, assign []int, lo, hi int) bool {
	changed := false
	for i := lo; i < hi; i++ {
		if c := nearest(i); c != assign[i] {
			assign[i] = c
			changed = true
		}
	}
	return changed
}

// Nearest returns the index of the L2-nearest centroid.
func Nearest(centroids [][]float32, v []float32) int {
	best, bestD := 0, float32(0)
	for i, c := range centroids {
		d := simd.L2Sq(c, v)
		if i == 0 || d < bestD {
			best, bestD = i, d
		}
	}
	return best
}

// farthest finds the unclaimed vector with the greatest distance to its
// assigned centroid — a deterministic reseed candidate for an emptied
// cluster. Claimed vectors are skipped so each reseed in a pass picks a
// distinct seed. Reseeds per pass ≤ k-1 ≤ n-1, so an unclaimed vector
// always exists.
func farthest(vecs [][]float32, centroids [][]float32, assign []int, claimed map[int]bool) int {
	best, bestD := -1, float32(-1)
	for i, v := range vecs {
		if claimed[i] {
			continue
		}
		if d := simd.L2Sq(centroids[assign[i]], v); best == -1 || d > bestD {
			best, bestD = i, d
		}
	}
	if best == -1 {
		return 0 // unreachable given the reseed bound; defensive
	}
	return best
}
