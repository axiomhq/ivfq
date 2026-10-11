package simd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// DotF32I8 agrees with the float64 sum at every tail length and for each
// kernel the CPU has; the float32 kernels may differ in the last bits.
func TestDotF32I8(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, k := range []struct{ avx512, avx2 bool }{{true, true}, {false, true}, {false, false}} {
		o512, o2 := hasDotF32I8AVX512, hasDotF32I8AVX2
		hasDotF32I8AVX512, hasDotF32I8AVX2 = o512 && k.avx512, o2 && k.avx2
		for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 257, 768, 1537} {
			x := make([]float32, n)
			b := make([]byte, n)
			var want, mag float64
			for i := range x {
				x[i] = float32(r.NormFloat64())
				b[i] = byte(r.IntN(256))
				if n == 257 {
					b[i] = 0x80 // -128 in every lane
				}
				p := float64(x[i]) * float64(int8(b[i]))
				want += p
				mag += math.Abs(p)
			}
			if got := DotF32I8(x, b); math.Abs(float64(got)-want) > 1e-5*mag+1e-6 {
				t.Fatalf("avx512=%v avx2=%v n=%d: %v, want %v", hasDotF32I8AVX512, hasDotF32I8AVX2, n, got, want)
			}
		}
		hasDotF32I8AVX512, hasDotF32I8AVX2 = o512, o2
	}
}

func BenchmarkDotF32I8(b *testing.B) {
	x := make([]float32, 768)
	row := make([]byte, 768)
	for b.Loop() {
		DotF32I8(x, row)
	}
}
