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
