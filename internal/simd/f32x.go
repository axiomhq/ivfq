package simd

import "math"

// Element-wise float32 kernels. Each result depends only on its own
// element, so the AVX2 kernels match the Go loops bit for bit.

// NonFinite returns the index of x's first NaN or infinite value, or -1.
func NonFinite(x []float32) int {
	if n8 := len(x) &^ 7; hasAVX2 && n8 > 0 && !nonFinite(&x[0], n8) {
		x = x[n8:]
		if i := nonFiniteGo(x); i >= 0 {
			return n8 + i
		}
		return -1
	}
	return nonFiniteGo(x)
}

func nonFiniteGo(x []float32) int {
	for i, v := range x {
		if math.Float32bits(v)&0x7f800000 == 0x7f800000 {
			return i
		}
	}
	return -1
}

// AddF64 adds float64(src[j]) to dst[j]. dst must be at least len(src) long.
func AddF64(dst []float64, src []float32) {
	dst = dst[:len(src)]
	n4 := len(src) &^ 3
	if hasAVX2 && n4 > 0 {
		addF64(&dst[0], &src[0], n4)
	} else {
		n4 = 0
	}
	for j := n4; j < len(src); j++ {
		dst[j] += float64(src[j])
	}
}

// ScaleF64 sets x[j] to float32(float64(x[j]) * s).
func ScaleF64(x []float32, s float64) {
	n4 := len(x) &^ 3
	if hasAVX2 && n4 > 0 {
		scaleF64(&x[0], n4, s)
	} else {
		n4 = 0
	}
	for j := n4; j < len(x); j++ {
		x[j] = float32(float64(x[j]) * s)
	}
}
