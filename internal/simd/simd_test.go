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

// NonFinite, AddF64 and ScaleF64 match their Go loops bit for bit at every
// tail length, and NonFinite finds the first bad value wherever it sits.
func TestF32xKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	old := hasAVX2
	defer func() { hasAVX2 = old }()
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 768, 771} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * math.Exp(r.NormFloat64()*20))
		}
		if n > 1 {
			x[0], x[n-1] = math.SmallestNonzeroFloat32, -math.MaxFloat32 // finite extremes
		}
		s := 1 / math.Sqrt(r.Float64()*1e3+1e-30)
		var want [2][]float32
		var sums [2][]float64
		for k, on := range []bool{true, false} {
			hasAVX2 = old && on
			if got := NonFinite(x); got != -1 {
				t.Fatalf("avx2=%v n=%d: NonFinite = %d on finite values", hasAVX2, n, got)
			}
			for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
				for _, at := range []int{0, n / 2, n - 1} {
					if at < 0 || at >= n {
						continue
					}
					y := append([]float32(nil), x...)
					y[at] = bad
					if n > 1 && at < n-1 {
						y[n-1] = float32(math.NaN()) // a later bad value must not win
					}
					if got := NonFinite(y); got != at {
						t.Fatalf("avx2=%v n=%d: NonFinite = %d, want %d", hasAVX2, n, got, at)
					}
				}
			}
			sums[k] = make([]float64, n+1)
			sums[k][n] = 7
			for i := range sums[k][:n] {
				sums[k][i] = r.NormFloat64()
			}
			if k == 1 {
				copy(sums[1], sums[0]) // same start for both
			}
			want[k] = append([]float32(nil), x...)
			ScaleF64(want[k], s)
		}
		// Recompute AddF64 from equal starts, then compare with a Go loop.
		base := append([]float64(nil), sums[0]...)
		for k, on := range []bool{true, false} {
			hasAVX2 = old && on
			copy(sums[k], base)
			AddF64(sums[k], x)
		}
		for j := range x {
			ref := base[j] + float64(x[j])
			if math.Float64bits(sums[0][j]) != math.Float64bits(ref) || math.Float64bits(sums[1][j]) != math.Float64bits(ref) {
				t.Fatalf("n=%d j=%d: AddF64 %v %v, want %v", n, j, sums[0][j], sums[1][j], ref)
			}
			ref32 := float32(float64(x[j]) * s)
			if math.Float32bits(want[0][j]) != math.Float32bits(ref32) || math.Float32bits(want[1][j]) != math.Float32bits(ref32) {
				t.Fatalf("n=%d j=%d: ScaleF64 %v %v, want %v", n, j, want[0][j], want[1][j], ref32)
			}
		}
		if sums[0][n] != 7 || sums[1][n] != 7 {
			t.Fatalf("n=%d: AddF64 wrote past len(src)", n)
		}
	}
}
