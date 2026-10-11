package simd

import "golang.org/x/sys/cpu"

var hasF32AVX512 = cpu.X86.HasAVX512F

// dotAVX512 is the AVX-512 inner product, len(a) > 0 and len(b) >= len(a).
func dotAVX512(a, b []float32) float32 {
	var s float32
	dotsF32AVX512(&a[0], &b[0], len(a), 1, &s)
	return s
}

//go:noescape
func dotsF32AVX512(q *float32, block *float32, dims, rows int, out *float32)

//go:noescape
func l2sqF32AVX512(a *float32, b *float32, n int) float32
