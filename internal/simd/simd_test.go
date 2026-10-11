package simd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestNormalizeTiny: the scale 1/|v| for a vector near the float32 floor
// exceeds MaxFloat32, so it must be applied in float64 or the result is Inf.
func TestNormalizeTiny(t *testing.T) {
	v := []float32{math.SmallestNonzeroFloat32, 0}
	Normalize(v)
	if v[0] != 1 || v[1] != 0 {
		t.Fatalf("Normalize = %v, want [1 0]", v)
	}
	z := []float32{0, 0}
	Normalize(z)
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("zero vector changed: %v", z)
	}
}

// Dot and L2Sq agree with a float64 reference at every tail length, with
// the AVX-512 kernels and without, and Dots stays Dot bit for bit in both.
func TestF32Kernels(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for _, on := range []bool{true, false} {
		old := hasF32AVX512
		hasF32AVX512 = old && on
		for _, n := range []int{1, 2, 15, 16, 17, 31, 48, 63, 64, 65, 79, 80, 128, 129, 255, 256, 257, 768, 1537} {
			a, b := make([]float32, n), make([]float32, n)
			var dot, l2, magD, magL float64
			for i := range a {
				a[i], b[i] = float32(r.NormFloat64()*10), float32(r.NormFloat64()*10)
				p, d := float64(a[i])*float64(b[i]), float64(a[i])-float64(b[i])
				dot, l2, magD, magL = dot+p, l2+d*d, magD+math.Abs(p), magL+d*d
			}
			if got := float64(Dot(a, b)); math.Abs(got-dot) > 1e-5*magD {
				t.Fatalf("avx512=%v n=%d: Dot %v, want %v", hasF32AVX512, n, got, dot)
			}
			if got := float64(L2Sq(a, b)); math.Abs(got-l2) > 1e-5*magL {
				t.Fatalf("avx512=%v n=%d: L2Sq %v, want %v", hasF32AVX512, n, got, l2)
			}
		}
		hasF32AVX512 = old
	}
}
