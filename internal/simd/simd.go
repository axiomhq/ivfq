// Package simd holds the numeric kernels every ivfq package scores with:
// the float32 distances, the block dot product the centroid tree ranks
// with, the 1-bit popcount products, the Givens rotation, and the int8
// codec. Each has a pure Go reference; amd64 picks an AVX2 kernel at run
// time when the CPU has it, and every kernel is pinned to its reference by
// a test.
package simd

import (
	"math"

	"github.com/go-simd/floats"
)

// Dot returns the inner product of a and b.
func Dot(a, b []float32) float32 { return floats.Float32Dot(a, b) }

// L2Sq returns the squared euclidean distance between a and b.
func L2Sq(a, b []float32) float32 {
	d := floats.Float32Distance(a, b)
	return d * d
}

// CosineSim returns the cosine similarity of a and b. A zero vector scores 0.
func CosineSim(a, b []float32) float32 {
	// floats returns NaN for a zero-magnitude input (its only NaN source
	// for finite input); that is this function's documented 0.
	if s := floats.Float32CosineSimilarity(a, b); s == s {
		return s
	}
	return 0
}

// Normalize scales v to unit length in place. A zero vector is left as is.
func Normalize(v []float32) {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return
	}
	s := 1 / math.Sqrt(n)
	for d := range v {
		v[d] = float32(float64(v[d]) * s)
	}
}
