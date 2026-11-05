package simd

//go:noescape
func decodeI8AVX2(dst *float32, src *byte, n int)

//go:noescape
func encodeI8AVX2(dst *byte, src *float32, n int) int
