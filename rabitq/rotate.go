package rabitq

import (
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/bits"
	"math/rand/v2"
	"sync"
)

// The rotation RaBitQ's error bound assumes: a random orthogonal transform,
// applied to every residual before it is sign-quantized. It is what makes
// the quantization error independent of the data's own axes — without it, a
// corpus whose energy sits in a few dimensions quantizes to a handful of
// distinct codes and the estimator's variance is whatever the corpus says
// it is, not O(1/sqrt(D)).
//
// A materialized D x D matrix is out of the question at embedding widths:
// building one is O(D^3), storing it is 9 MB at 1536 dims, and applying it
// to every row of every cluster is D^2 per vector. So the rotation is a few
// ROUNDS of a random permutation, random signs and a Walsh-Hadamard
// transform (the construction of fast randomized Hadamard rotations). D is
// split into blocks by its binary digits (1536 = 1024 + 512, 100 = 64 + 32 +
// 4) and each block gets its own normalized Hadamard transform, so every
// factor is exactly orthogonal at any D: no power-of-two padding and no code
// bits spent on a pad. The permutation moves coordinates between blocks
// every round. It costs O(D log D) to apply, runs in vector registers, and
// is reproducible from a seed: a column stores the seed, not the matrix.
// The coverage test (TestBitBoundCoversAtItsConfidence, including its
// axis-aligned corpus) is what holds the round count honest.
//
// A column written with another construction under the same seed decodes
// to garbage: re-encode columns when this changes.
//
// Rotation is the transform RaBitQ applies.
type Rotation struct {
	seed   uint64
	dims   int
	blocks []int       // power-of-two block sizes, summing to dims
	perm   [][]int32   // per round
	coef   [][]float32 // coef[r][i]: random sign times the previous block's 1/sqrt(size)
	final  []float32   // the last block's 1/sqrt(size) at each position
	ident  []int32     // 0..dims-1, for the closing scale of a block
	pool   sync.Pool   // *[]float32 scratch for Apply
}

// rotationRounds is odd, so the rounds (one pass each, plus the closing
// scale pass) end in the caller's buffer.
const rotationRounds = 3

// NewRotation derives the rotation for dims-wide vectors from a seed.
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
	scale := make([]float32, dims) // scale[p]: 1/sqrt(size of the block holding position p)
	for rest, off := dims, 0; rest > 0; {
		s := 1 << (bits.Len(uint(rest)) - 1)
		r.blocks = append(r.blocks, s)
		for p := off; p < off+s; p++ {
			scale[p] = float32(1 / math.Sqrt(float64(s)))
		}
		rest, off = rest-s, off+s
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	prev := make([]float32, dims) // the scale of what the round before left at each position
	for i := range prev {
		prev[i] = 1
	}
	for round := 0; round < rotationRounds; round++ {
		perm := make([]int32, dims)
		for i := range perm {
			perm[i] = int32(i)
		}
		for i := dims - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			perm[i], perm[j] = perm[j], perm[i]
		}
		coef := make([]float32, dims)
		for i, p := range perm {
			coef[i] = prev[p]
			if rng.Uint64()&1 == 1 {
				coef[i] = -coef[i]
			}
		}
		r.perm, r.coef = append(r.perm, perm), append(r.coef, coef)
		prev = scale
	}
	r.final = scale
	r.ident = make([]int32, dims)
	for i := range r.ident {
		r.ident[i] = int32(i)
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
	if r.dims <= 1 {
		return
	}
	sp, _ := r.pool.Get().(*[]float32)
	if sp == nil {
		sp = new([]float32)
	}
	if len(*sp) < r.dims {
		*sp = make([]float32, r.dims)
	}
	src, dst := v[:r.dims], (*sp)[:r.dims]
	for round := range r.perm {
		simd.PermScale(dst, src, r.perm[round], r.coef[round])
		off := 0
		for _, s := range r.blocks {
			simd.FWHT(dst[off : off+s])
			off += s
		}
		src, dst = dst, src
	}
	simd.Scale(dst, src, r.final)
	r.pool.Put(sp)
}

// applyBlock rotates a dimension-major block of simd.FillBlock rows
// (block[j*simd.FillBlock+r] is dim j of row r): Apply on each row, bit for
// bit, across the block at once. block must be r.dims*simd.FillBlock long.
func (r *Rotation) applyBlock(block []float32) {
	if len(block) != r.dims*simd.FillBlock {
		panic("rabitq: rotation block of the wrong size")
	}
	if r.dims <= 1 {
		return
	}
	sp, _ := r.pool.Get().(*[]float32)
	if sp == nil {
		sp = new([]float32)
	}
	if len(*sp) < len(block) {
		*sp = make([]float32, len(block))
	}
	src, dst := block, (*sp)[:len(block)]
	for round := range r.perm {
		simd.PermScaleRows(dst, src, r.perm[round], r.coef[round])
		off := 0
		for _, s := range r.blocks {
			simd.FWHTRows(dst[off*simd.FillBlock:(off+s)*simd.FillBlock], s)
			off += s
		}
		src, dst = dst, src
	}
	simd.PermScaleRows(dst, src, r.ident, r.final)
	r.pool.Put(sp)
}

// Seed derives a rotation seed from names: FNV-1a over parts joined by
// '/', an index and field name say. Deriving it rather than drawing it
// keeps every column of a field in ONE frame, so one Rotation serves them
// all, while two different fields still get independent rotations. The
// value is written into every column anyway, so a reader never re-derives
// it and changing this function cannot strand existing columns.
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
