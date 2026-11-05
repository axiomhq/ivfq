package simd

//go:noescape
func dotsF32AVX512(q *float32, block *float32, dims, rows int, out *float32)

//go:noescape
func l2sqF32AVX512(a *float32, b *float32, n int) float32
