package simd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Dots is Dot per row, bit for bit, at every width's tail, with the
// AVX-512 kernels and without (then Dots runs its AVX2 kernel and Dot
// go-simd's).
func TestDotsMatchesDot(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 2))
	old := hasAVX512F
	defer func() { hasAVX512F = old }()
	for _, on := range []bool{true, false} {
		hasAVX512F = old && on
		for _, dims := range []int{1, 3, 7, 8, 9, 15, 16, 17, 100, 128, 129, 300} {
			for _, rows := range []int{0, 1, 2, 3, 4, 5, 7, 100} {
				q := make([]float32, dims)
				for i := range q {
					q[i] = float32(rng.NormFloat64() * 100)
				}
				block := make([]float32, dims*rows)
				for i := range block {
					block[i] = float32(rng.NormFloat64() * 100)
				}
				out := make([]float32, rows)
				Dots(q, block, out)
				for r := range out {
					if want := Dot(q, block[r*dims:(r+1)*dims]); math.Float32bits(out[r]) != math.Float32bits(want) {
						t.Fatalf("avx512=%v dims=%d row %d: %v, Dot %v", hasAVX512F, dims, r, out[r], want)
					}
				}
			}
		}
	}
}

// Dots3 is each row's dot with each block row, within float rounding of a
// float64 sum, and a pair's value does not depend on the rows beside it:
// a row against a block equals that row in any slot of any call.
func TestDots3(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 3))
	for _, avx := range []bool{true, false} {
		old := hasAVX2FMA
		hasAVX2FMA = old && avx
		for _, dims := range []int{1, 7, 8, 16, 24, 100, 768} {
			for _, rows := range []int{0, 1, 3, 4, 5, 9, 42} {
				vec := func(n int) []float32 {
					v := make([]float32, n)
					for i := range v {
						v[i] = float32(rng.NormFloat64())
					}
					return v
				}
				a, b, c, block := vec(dims), vec(dims), vec(dims), vec(dims*rows)
				out, alone := make([]float32, 3*rows), make([]float32, 3*rows)
				Dots3(a, b, c, block, out)
				for k, row := range [][]float32{a, b, c} {
					Dots3(vec(dims), row, vec(dims), block, alone)
					for r := range rows {
						var want float64
						for i, x := range row {
							want += float64(x) * float64(block[r*dims+i])
						}
						got := out[3*r+k]
						if math.Abs(float64(got)-want) > 1e-4*math.Sqrt(float64(dims)) {
							t.Fatalf("avx=%v dims=%d rows=%d row %d,%d: %v, want %v", avx, dims, rows, k, r, got, want)
						}
						if alone[3*r+1] != got {
							t.Fatalf("avx=%v dims=%d rows=%d row %d,%d: %v in slot %d, %v in slot 1", avx, dims, rows, k, r, got, k, alone[3*r+1])
						}
					}
				}
			}
		}
		hasAVX2FMA = old
	}
}
