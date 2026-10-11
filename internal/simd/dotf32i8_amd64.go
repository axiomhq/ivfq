package simd

import "golang.org/x/sys/cpu"

var (
	hasDotF32I8AVX512 = cpu.X86.HasAVX512F
	hasDotF32I8AVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
)

func dotF32I8(x []float32, b []byte) float32 {
	switch {
	case hasDotF32I8AVX512:
		return dotF32I8AVX512(&x[0], &b[0], len(x))
	case hasDotF32I8AVX2:
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

//go:noescape
func dotF32I8AVX512(x *float32, b *byte, n int) float32

//go:noescape
func dotF32I8AVX2(x *float32, b *byte, n int) float32
