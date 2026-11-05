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
// the AVX-512 kernels and without.
func TestF32Kernels(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	old := hasAVX512F
	defer func() { hasAVX512F = old }()
	for _, on := range []bool{true, false} {
		hasAVX512F = old && on
		for _, n := range []int{1, 2, 15, 16, 17, 31, 48, 63, 64, 65, 79, 80, 128, 129, 255, 256, 257, 768, 1537} {
			a, b := make([]float32, n), make([]float32, n)
			var dot, l2, magD, magL float64
			for i := range a {
				a[i], b[i] = float32(r.NormFloat64()*10), float32(r.NormFloat64()*10)
				p, d := float64(a[i])*float64(b[i]), float64(a[i])-float64(b[i])
				dot, l2, magD, magL = dot+p, l2+d*d, magD+math.Abs(p), magL+d*d
			}
			if got := float64(Dot(a, b)); math.Abs(got-dot) > 1e-5*magD {
				t.Fatalf("avx512=%v n=%d: Dot %v, want %v", hasAVX512F, n, got, dot)
			}
			if got := float64(L2Sq(a, b)); math.Abs(got-l2) > 1e-5*magL {
				t.Fatalf("avx512=%v n=%d: L2Sq %v, want %v", hasAVX512F, n, got, l2)
			}
		}
	}
}

// A broken length precondition panics with and without the vector kernels,
// instead of reading or writing outside the slices. The short slices keep
// spare capacity, which a slice-expression guard would accept.
func TestShapeMismatchPanics(t *testing.T) {
	old := hasAVX512F
	defer func() { hasAVX512F = old }()
	for _, on := range []bool{true, false} {
		hasAVX512F = old && on
		for name, f := range map[string]func(){
			"Dot":             func() { Dot(make([]float32, 16), make([]float32, 17)) },
			"L2Sq":            func() { L2Sq(make([]float32, 16), make([]float32, 17)) },
			"FWHT":            func() { FWHT(make([]float32, 24)) },
			"FWHTRows":        func() { FWHTRows(make([]float32, 3*FillBlock), 3) },
			"DecodeI8":        func() { DecodeI8(make([]float32, 16), make([]byte, 8, 16)) },
			"BitProductBytes": func() { BitProductBytes(make([]byte, 8, 16), make([]uint64, 8), 2) },
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("avx512=%v: %s with a bad shape did not panic", hasAVX512F, name)
					}
				}()
				f()
			}()
		}
	}
}

// CosineSim scores 0, never NaN or ±Inf, when a norm is 0, underflows or
// overflows float32.
func TestCosineSimDegenerate(t *testing.T) {
	for _, c := range []struct{ a, b []float32 }{
		{[]float32{0, 0}, []float32{1, 0}},
		{[]float32{1e-30, 0}, []float32{1, 0}},
		{[]float32{1e20, 1e20}, []float32{1e20, 1e20}},
		{[]float32{1e20, 1e20}, []float32{1, 1}},
	} {
		if got := CosineSim(c.a, c.b); got != 0 {
			t.Errorf("CosineSim(%v, %v) = %v, want 0", c.a, c.b, got)
		}
	}
	if got := CosineSim([]float32{3, 4}, []float32{6, 8}); math.Abs(float64(got)-1) > 1e-6 {
		t.Errorf("CosineSim of parallel vectors = %v, want 1", got)
	}
}
