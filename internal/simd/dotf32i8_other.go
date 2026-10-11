//go:build !amd64

package simd

func dotF32I8(x []float32, b []byte) float32 { return dotF32I8Generic(x, b) }

var hasDotF32I8AVX512, hasDotF32I8AVX2 bool
