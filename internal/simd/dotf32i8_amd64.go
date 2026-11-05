package simd

//go:noescape
func dotF32I8AVX512(x *float32, b *byte, n int) float32

//go:noescape
func dotF32I8AVX2(x *float32, b *byte, n int) float32
