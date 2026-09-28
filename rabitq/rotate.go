package rabitq

import (
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/bits"
	"math/rand/v2"
)

// The rotation RaBitQ's error bound assumes: a uniformly random orthogonal
// transform, applied to every residual before it is sign-quantized. It is
// what makes the quantization error independent of the data's own axes —
// without it, a corpus whose energy sits in a few dimensions quantizes to a
// handful of distinct codes and the estimator's variance is whatever the
// corpus says it is, not O(1/sqrt(D)).
//
// A materialized D x D matrix is out of the question at the dimensions this
// engine serves: building one is O(D^3) (3.6 GFLOP at 1536 dims), storing it
// is 9 MB, and applying it to every row of every hood at build time is
// D^2 per vector. So the rotation is a BUTTERFLY of Givens rotations: a few
// rounds, each one a random pairing of all D coordinates and a random angle
// per pair. Every factor is exactly orthogonal, so the product is exactly
// orthogonal at any D (no power-of-two padding, no code bits spent on a pad),
// it costs O(D log D) to apply, and it is reproducible from a seed — the pack
// stores the seed, not the matrix.
//
// rounds is 3 + ceil(log2(D)): each round mixes disjoint pairs, so after
// log2(D) rounds of random pairings every coordinate has had a path to every
// other, and the three extra rounds are the mixing margin. The coverage test
// (TestBitBoundCoversAtItsConfidence, including its axis-aligned corpus) is
// what actually holds this number honest.
// Rotation is the butterfly of Givens rotations RaBitQ applies.
type Rotation struct {
	seed  uint64
	dims  int
	pairs []simd.Pair
}

func rotationRounds(dims int) int {
	if dims <= 1 {
		return 0
	}
	return 3 + bits.Len(uint(dims-1))
}

// NewRotation derives the butterfly for dims-wide vectors from a seed.
// Both the encoder and every reader build the same one from the seed a
// column records, so the frame a column's bits live in is a property of
// the bytes, not of the process. A rotation is immutable and safe to share;
// a caller builds one per (seed, dims) it serves and passes it to
// Quantize, AppendRows and NewQuery.
func NewRotation(seed uint64, dims int) *Rotation {
	r := &Rotation{seed: seed, dims: dims}
	if dims <= 1 {
		return r
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	perm := make([]int32, dims)
	rounds := rotationRounds(dims)
	r.pairs = make([]simd.Pair, 0, rounds*(dims/2))
	for round := 0; round < rounds; round++ {
		for i := range perm {
			perm[i] = int32(i)
		}
		for i := dims - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			perm[i], perm[j] = perm[j], perm[i]
		}
		for i := 0; i+1 < dims; i += 2 {
			theta := rng.Float64() * 2 * math.Pi
			r.pairs = append(r.pairs, simd.Pair{I: perm[i], J: perm[i+1],
				Cs: float32(math.Cos(theta)), Sn: float32(math.Sin(theta))})
		}
	}
	return r
}

// Seed is the seed r was derived from.
func (r *Rotation) Seed() uint64 { return r.seed }

// Dims is the width of the vectors r rotates.
func (r *Rotation) Dims() int { return r.dims }

// Apply rotates v in place. v must be r.dims long.
func (r *Rotation) Apply(v []float32) {
	_ = v[r.dims-1]
	for _, p := range r.pairs {
		a, b := v[p.I], v[p.J]
		v[p.I], v[p.J] = p.Cs*a-p.Sn*b, p.Sn*a+p.Cs*b
	}
}

// applyBlock rotates a dimension-major block of simd.FillBlock rows
// (block[j*simd.FillBlock+r] is dim j of row r): Apply on each row, pair by
// pair across the block. block must be r.dims*simd.FillBlock long.
func (r *Rotation) applyBlock(block []float32) {
	if len(block) != r.dims*simd.FillBlock {
		panic("rabitq: rotation block of the wrong size")
	}
	if len(r.pairs) > 0 {
		simd.RotatePairs(r.pairs, block)
	}
}

// Seed is the rotation seed a namespace's field uses: FNV-1a over the
// namespace and field names. Deriving it rather than drawing it keeps every
// hood of a field in ONE frame, so a query rotates once per hood and the
// cache above holds one entry — while two different fields, and two
// different namespaces, still get independent rotations. The value is
// written into every codes part anyway, so a reader never re-derives it and
// changing this function cannot strand existing packs.
func Seed(parts ...string) uint64 {
	const offset, prime = 14695981039346656037, 1099511628211
	h := uint64(offset)
	for i, s := range parts {
		if i > 0 {
			h = (h ^ '/') * prime
		}
		for j := 0; j < len(s); j++ {
			h = (h ^ uint64(s[j])) * prime
		}
	}
	if h == 0 {
		h = prime // 0 is the "unseeded" value the decoder rejects
	}
	return h
}
