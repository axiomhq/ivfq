package rabitq

import (
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/rand/v2"
	"testing"
)

func TestSeedNeverZero(t *testing.T) {
	if Seed() == 0 {
		t.Fatal("empty seed")
	}
	if Seed("ns", "vector") == 0 {
		t.Fatal("named seed")
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5}
	var n0 float64
	for _, x := range v {
		n0 += float64(x) * float64(x)
	}
	NewRotation(Seed("t", "r"), len(v)).Apply(v)
	var n1 float64
	for _, x := range v {
		n1 += float64(x) * float64(x)
	}
	if math.Abs(n0-n1) > 1e-5 {
		t.Fatalf("norm %v -> %v", n0, n1)
	}
}

func TestRotatePairsMatchesApply(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, dims := range []int{2, 17, 128} {
		rot := NewRotation(99, dims)
		rows := make([][]float32, simd.FillBlock)
		block := make([]float32, dims*simd.FillBlock)
		for r := range rows {
			rows[r] = make([]float32, dims)
			for j := range rows[r] {
				rows[r][j] = float32(rng.NormFloat64())
				block[j*simd.FillBlock+r] = rows[r][j]
			}
		}
		rot.applyBlock(block)
		for r, row := range rows {
			rot.Apply(row)
			for j, x := range row {
				if math.Float32bits(block[j*simd.FillBlock+r]) != math.Float32bits(x) {
					t.Fatalf("dims=%d row %d dim %d: block %v, Apply %v", dims, r, j, block[j*simd.FillBlock+r], x)
				}
			}
		}
	}
}
