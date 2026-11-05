package simd

//go:noescape
func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32)
