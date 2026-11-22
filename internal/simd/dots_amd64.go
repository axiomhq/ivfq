package simd

//go:noescape
func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32)

//go:noescape
func dots3AVX2FMA(a, b, c, block *float32, dims, rows int, out *float32)
