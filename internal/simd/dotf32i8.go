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
	return dotF32I8(x, b)
}

func dotF32I8Generic(x []float32, b []byte) float32 {
	var s float32
	for i, v := range x {
		s += v * float32(int8(b[i]))
	}
	return s
}
