//go:build !amd64

package simd

var hasF32AVX512 bool

func dotAVX512(a, b []float32) float32 { panic("unreachable") }

func l2sqF32AVX512(a *float32, b *float32, n int) float32 { panic("unreachable") }
