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
// Coordinate i accumulates in float32 lane i&63, including the final partial
// block. Products round before addition (no FMA). The 64 lanes fold to 16 as
// (lane[j]+lane[j+16])+(lane[j+32]+lane[j+48]), then by offsets 8, 4, 2, 1.
// Go, AVX2 and AVX-512 use this order, without a square root.
func L2Sq(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("ivfq: L2Sq: slice length mismatch")
	}
	if len(a) > 0 {
		if hasAVX512F {
			return l2sqF32AVX512(&a[0], &b[0], len(a))
		}
		if hasAVX2 {
			return l2sqF32AVX2(&a[0], &b[0], len(a))
		}
	}
	var lanes [64]float32
	for i := range a {
		d := a[i] - b[i]
		lanes[i&63] += float32(d * d)
	}
	for j := range 16 {
		lanes[j] = (lanes[j] + lanes[j+16]) + (lanes[j+32] + lanes[j+48])
	}
	for offset := 8; offset > 0; offset /= 2 {
		for j := range offset {
			lanes[j] += lanes[j+offset]
		}
	}
	return lanes[0]
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
