package simd

//go:noescape
func dotsF32AVX512(q *float32, block *float32, dims, rows int, out *float32)

// L2Sq's 64-lane float32 kernels, for n > 0.
//
//go:noescape
func l2sqF32AVX512(a *float32, b *float32, n int) float32

//go:noescape
func l2sqF32AVX2(a *float32, b *float32, n int) float32

// nonFinite is NonFinite's test for n a multiple of 8, n >= 8.
//
//go:noescape
func nonFinite(x *float32, n int) bool

// addF64 is AddF64 for n a multiple of 4, n >= 4.
//
//go:noescape
func addF64(dst *float64, src *float32, n int)

// addF32 is AddF32 for n a multiple of 8, n >= 8.
//
//go:noescape
func addF32(dst *float32, src *float32, n int)

// scaleF64 is ScaleF64 for n a multiple of 4, n >= 4.
//
//go:noescape
func scaleF64(x *float32, n int, s float64)
