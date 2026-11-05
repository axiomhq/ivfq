package simd

// DotF32I8 returns the inner product of x and the int8 row b (b[i] read as
// int8), len(x) long: DecodeI8 and Dot fused, with no float32 scratch row.
// The vector kernels sum in a different order than Dot, so the result
// differs from Dot(x, decoded) in the last bits.
func DotF32I8(x []float32, b []byte) float32 {
	if len(x) == 0 {
		return 0
	}
	_ = b[len(x)-1]
	switch {
	case hasAVX512F:
		return dotF32I8AVX512(&x[0], &b[0], len(x))
	case hasAVX2FMA:
		n := len(x) &^ 7
		var s float32
		if n > 0 {
			s = dotF32I8AVX2(&x[0], &b[0], n)
		}
		for i := n; i < len(x); i++ {
			s += x[i] * float32(int8(b[i]))
		}
		return s
	}
	return dotF32I8Generic(x, b)
}

func dotF32I8Generic(x []float32, b []byte) float32 {
	var s float32
	for i, v := range x {
		s += v * float32(int8(b[i]))
	}
	return s
}
