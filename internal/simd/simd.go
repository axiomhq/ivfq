// Package simd holds the numeric kernels every ivfq package scores with:
// the float32 distances, the block dot product the centroid tree ranks
// with, the 1-bit popcount products, the Walsh-Hadamard rotation and the
// quantizers behind rabitq, and the int8 codec. Each has a pure Go
// reference; amd64 picks an AVX2 or AVX-512 kernel at run time when the CPU
// has it, and every kernel is pinned to its reference by a test.
package simd

import (
	"math"

	"github.com/go-simd/floats"
)

// Dot returns the inner product of a and b.
func Dot(a, b []float32) float32 {
	if hasAVX512F && len(a) > 0 && len(b) == len(a) {
		var s float32
		dotsF32AVX512(&a[0], &b[0], len(a), 1, &s)
		return s
	}
	return floats.Float32Dot(a, b)
}

// L2Sq returns the squared euclidean distance between a and b.
func L2Sq(a, b []float32) float32 {
	if hasAVX512F && len(a) > 0 && len(b) == len(a) {
		return l2sqF32AVX512(&a[0], &b[0], len(a))
	}
	d := floats.Float32Distance(a, b)
	return d * d
}

// CosineSim returns the cosine similarity of a and b. A zero vector, or one
// whose float32 squared norm rounds to 0 or overflows, scores 0; with
// elements below about 1e-19 the result loses precision.
func CosineSim(a, b []float32) float32 {
	// floats returns NaN or ±Inf for those (0/0, x/0, Inf/Inf).
	s := float64(floats.Float32CosineSimilarity(a, b))
	if math.IsNaN(s) || math.IsInf(s, 0) {
		return 0
	}
	return float32(s)
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
