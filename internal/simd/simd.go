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
	if len(a) != len(b) {
		panic("ivfq: L2Sq: slice length mismatch")
	}
	// Match SumSq's four float64 lanes on every CPU, without a sqrt or FMA.
	n4 := len(a) &^ 3
	var s float64
	if hasAVX2FMA && n4 > 0 {
		s = l2sq(&a[0], &b[0], n4)
	} else {
		var lanes [4]float64
		for i := 0; i < n4; i++ {
			d := float64(a[i]) - float64(b[i])
			lanes[i&3] += float64(d * d)
		}
		s = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
	}
	for i := n4; i < len(a); i++ {
		d := float64(a[i]) - float64(b[i])
		s += float64(d * d)
	}
	return float32(s)
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
// The squared norm is SumSq's, in its lane order on every CPU.
func Normalize(v []float32) {
	n := SumSq(v)
	if n == 0 {
		return
	}
	ScaleF64(v, 1/math.Sqrt(n))
}
