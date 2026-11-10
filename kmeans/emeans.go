package kmeans

import (
	"cmp"
	"context"
	"math"
	"math/rand"
	"slices"

	"github.com/axiomhq/ivfq/internal/simd"
)

// EMeansIters is the pass count FitEMeans was measured at:
// Config{K: k, Iters: EMeansIters}.FitEMeans.
const EMeansIters = 5

// FitEMeans is Fit with E-means training in place of k-means++ and Lloyd's:
// the same sample, the same parallel assignment of every vector, and the
// same rules for vecs, an empty corpus, a cancelled ctx and GOMAXPROCS.
// Iters counts passes over the sample; EMeansIters is the measured default.
//
// E-means (Faiss contrib/e_means) starts from K random sample rows and
// moves the centroids after every mini-batch instead of every pass, so it
// pays for neither the seeding nor a full pass per step. On SIFT1M
// prefixes (128 dims, 8 cores, K = ivfq.HoodK) the whole fit measured
// 0.19 s against Fit's 1.58 s at 100k rows and 19 ms against 43 ms at 10k,
// with 0.4% less squared error. A cluster the assignment leaves empty takes
// half of the cluster with the most squared error, split as FitBalanced
// splits one; none was empty in those runs.
//
// What it gives up is the seeding: a random start does not spread over
// well-separated clusters the way k-means++ does. On gaussian clusters a
// tenth as wide as their spacing, with K their count, it measured 4 to 18
// times Fit's squared error. On overlapping ones it stayed within 2%.
//
// Above TwoLevelK it is Fit itself, Iters counting Lloyd iterations: the
// two-level fit spends its time elsewhere and measured no faster with
// E-means inside.
func (c Config) FitEMeans(ctx context.Context, vecs [][]float32) ([][]float32, []int, error) {
	n := len(vecs)
	if n == 0 {
		return nil, nil, nil
	}
	k := min(max(c.K, 1), n)
	if k > TwoLevelK {
		return c.Fit(ctx, vecs)
	}
	train := vecs
	if s := sampleSizeForBudget(k, len(vecs[0]), c.MaxBytes); s < n {
		var err error
		if train, err = sample(ctx, vecs, s, c.Seed); err != nil {
			return nil, nil, err
		}
	}
	centroids, err := emeans(ctx, train, k, c.Iters, c.Seed, c.Spherical)
	if err != nil {
		return nil, nil, err
	}
	assign := make([]int, n)
	assignAll(ctx, centroids, vecs, assign)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	// A cluster the closing means left empty takes half of the cluster with
	// the most squared error, split as FitBalanced splits one. Lloyd's
	// reseed, onto the farthest vector, would win that vector alone
	// wherever no other is nearer to it than to its own centroid, which in
	// many dimensions is most of the time.
	sizes := make([]int, len(centroids))
	for _, a := range assign {
		sizes[a]++
	}
	if !slices.Contains(sizes, 0) {
		return centroids, assign, nil
	}
	sse := make([]float64, len(centroids))
	for i, a := range assign {
		sse[a] += float64(simd.L2Sq(centroids[a], vecs[i]))
	}
	var moved []int
	for empty, m := range sizes {
		if m > 0 {
			continue
		}
		donor := -1
		for d, e := range sse {
			if sizes[d] > 1 && e > 0 && (donor < 0 || e > sse[donor]) {
				donor = d
			}
		}
		if donor < 0 {
			break // every vector sits on its centroid: nothing to split
		}
		var rows []int
		var members [][]float32
		for i, a := range assign {
			if a == donor {
				rows, members = append(rows, i), append(members, vecs[i])
			}
		}
		halves, side, err := Config{K: 2, Iters: SplitIters, Seed: c.Seed, Spherical: c.Spherical}.FitBalanced(ctx, members)
		if err != nil {
			return nil, nil, err
		}
		pair := [2]int{donor, empty}
		for h, to := range pair {
			centroids[to], sizes[to], sse[to] = halves[h], 0, 0
		}
		for j, i := range rows {
			to := pair[side[j]]
			assign[i] = to
			sizes[to]++
			sse[to] += float64(simd.L2Sq(centroids[to], vecs[i]))
		}
		moved = append(moved, donor, empty)
	}
	// Only the split centroids moved. A vector in one of those clusters is
	// compared against every centroid; any other has for nearest its old
	// centroid or a moved one, the lower index on a tie as in Nearest. That
	// is the full pass again, at 1+len(moved) distances for most vectors.
	assignWith(ctx, n, assign, 1+len(moved), func(i int) int {
		best := assign[i]
		if slices.Contains(moved, best) {
			return Nearest(centroids, vecs[i])
		}
		bestD := simd.L2Sq(centroids[best], vecs[i])
		for _, s := range moved {
			if d := simd.L2Sq(centroids[s], vecs[i]); d < bestD || d == bestD && s < best {
				best, bestD = s, d
			}
		}
		return best
	})
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return centroids, assign, nil
}

// emeans is the NumPy reference of Faiss contrib/e_means ("E-Means: Scaling
// Laws for Mini-Batch K-Means", Algorithm 1) over all of vecs, which must be
// non-empty, with k >= 1; the constants are that implementation's.
//
// Each centroid is a moving average of the batch means it won, its step
// weighted by usage, the decayed count of rows it has won so far. The batch
// size and the first step come from the fitted laws below, the step anneals
// by a cosine to a tenth, and every fifth batch revive splits a busy
// centroid into each nearly unused one. The centroids returned are the
// means of the rows each won over the last pass; one that won none keeps
// its position.
//
// Spherical follows the reference too: rows are assigned to the centroids'
// directions, the averages themselves stay unnormalized, and the result is
// normalized.
//
// The batch assignment fans out as assignAll does and every sum is serial,
// so the result is bit-identical whatever GOMAXPROCS is. ctx is checked
// once a batch.
func emeans(ctx context.Context, vecs [][]float32, k, epochs int, seed int64, spherical bool) ([][]float32, error) {
	n, dims := len(vecs), len(vecs[0])
	k = min(k, n) // a MaxBytes sample can hold fewer rows than k
	rng := rand.New(rand.NewSource(seed))
	cents := make([][]float32, k)
	for c, i := range rng.Perm(n)[:k] {
		cents[c] = slices.Clone(vecs[i])
		if spherical {
			simd.Normalize(cents[c])
		}
	}
	if epochs < 1 {
		return cents, nil
	}
	near := cents // what a row is assigned against
	if spherical {
		near = make([][]float32, k)
		for c := range near {
			near[c] = make([]float32, dims)
		}
	}

	// Batch law: at 16 dimensions or fewer the reference makes 8 times as
	// many, smaller updates, which is where the larger batch cost it quality.
	fn, fk, ft := float64(n), float64(k), float64(epochs)
	coef, limit := 0.035134, roundPow2(fn/2)
	if dims <= 16 {
		coef, limit = 0.00439175, n/2
	}
	batch := roundPow2(coef * math.Pow(fk, 0.134749) * math.Pow(fn, 0.876473) * math.Pow(ft, 0.813344))
	batch = max(1, min(batch, limit, n))
	// Step law, for that batch.
	alpha0 := 0.8244 * math.Pow(fk, -0.4617) * math.Pow(float64(batch), 0.9003) * math.Pow(fn, -0.4498) * math.Pow(ft, -0.1057)
	alpha0 = min(max(alpha0, 1e-4), 1)
	steps := epochs * ((n + batch - 1) / batch)

	usage := make([]float64, k)
	for c := range usage {
		usage[c] = 1.0 / 65536
	}
	sums := make([]float64, k*dims) // row c is centroid c's sum over this batch
	counts := make([]int, k)
	lastSums := make([]float64, k*dims) // the same over the whole last pass
	lastCounts := make([]int, k)
	label := make([]int, batch)
	step := 0
	for e := range epochs {
		order := rng.Perm(n)
		for lo := 0; lo < n; lo += batch {
			rows := order[lo:min(lo+batch, n)]
			if spherical {
				for c := range cents {
					copy(near[c], cents[c])
					simd.Normalize(near[c])
				}
			}
			assignWith(ctx, len(rows), label, k, func(i int) int { return Nearest(near, vecs[rows[i]]) })
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			clear(sums)
			clear(counts)
			for i, r := range rows {
				c := label[i]
				counts[c]++
				row := sums[c*dims : (c+1)*dims]
				for d, x := range vecs[r] {
					row[d] += float64(x)
				}
			}
			if e == epochs-1 {
				for i, s := range sums {
					lastSums[i] += s
				}
				for c, m := range counts {
					lastCounts[c] += m
				}
			}
			alpha := alpha0 * (0.1 + 0.9*0.5*(1+math.Cos(math.Pi*float64(step)/float64(steps))))
			for c := range k {
				usage[c] *= 1 - alpha
				if counts[c] == 0 {
					continue
				}
				m := float64(counts[c])
				usage[c] += alpha * m
				w := alpha * m / usage[c]
				row := sums[c*dims : (c+1)*dims]
				for d, cur := range cents[c] {
					cents[c][d] = float32(float64(cur) + w*(row[d]/m-float64(cur)))
				}
			}
			// The reference leaves revival off in 2 dimensions or fewer,
			// where it measured a higher held-out error with it.
			if step++; step%5 == 0 && dims > 2 {
				revive(cents, usage, rng)
			}
		}
	}
	for c := range k {
		if lastCounts[c] > 0 {
			for d := range cents[c] {
				cents[c][d] = float32(lastSums[c*dims+d] / float64(lastCounts[c]))
			}
		}
		if spherical {
			simd.Normalize(cents[c])
		}
	}
	return cents, nil
}

// revive gives every centroid whose usage is under a tenth of the mean a
// place next to a busy one: donors are drawn without replacement in
// proportion to usage (the largest Gumbel keys), and donor and child move
// 1e-4 of each coordinate apart, in alternating directions, and halve the
// donor's usage between them.
func revive(cents [][]float32, usage []float64, rng *rand.Rand) {
	var sum float64
	for _, u := range usage {
		sum += u
	}
	unusedBelow := 0.1 * sum / float64(len(usage))
	var unused, donors []int
	for c, u := range usage {
		if u < unusedBelow {
			unused = append(unused, c)
		} else {
			donors = append(donors, c)
		}
	}
	if len(unused) == 0 {
		return
	}
	keys := make([]float64, len(usage))
	for _, c := range donors { // usage >= unusedBelow > 0
		keys[c] = math.Log(usage[c]) - math.Log(-math.Log(max(rng.Float64(), math.SmallestNonzeroFloat64)))
	}
	slices.SortFunc(donors, func(a, b int) int { return cmp.Compare(keys[b], keys[a]) })
	for i := range min(len(unused), len(donors)) {
		from, to := donors[i], unused[i]
		for d, x := range cents[from] {
			off := float32(1e-4)
			if d%2 == 1 {
				off = -off
			}
			cents[from][d], cents[to][d] = x*(1-off), x*(1+off)
		}
		usage[from] /= 2
		usage[to] = usage[from]
	}
}

// roundPow2 rounds a positive value to the nearest power of two, by ratio.
func roundPow2(v float64) int {
	if v <= 1 {
		return 1
	}
	low := math.Pow(2, math.Floor(math.Log2(v)))
	if v/low < 2*low/v {
		return int(low)
	}
	return int(2 * low)
}
