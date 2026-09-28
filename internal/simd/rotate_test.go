package simd

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// The dispatched kernel is rotatePairsGeneric bit for bit, at every width
// including ones with an odd coordinate left unpaired.
func TestRotatePairsMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, dims := range []int{2, 3, 17, 128} {
		var pairs []Pair
		for round := 0; round < 4; round++ {
			perm := rng.Perm(dims)
			for i := 0; i+1 < dims; i += 2 {
				theta := rng.Float64() * 2 * math.Pi
				pairs = append(pairs, Pair{I: int32(perm[i]), J: int32(perm[i+1]), Cs: float32(math.Cos(theta)), Sn: float32(math.Sin(theta))})
			}
		}
		block := make([]float32, dims*FillBlock)
		for i := range block {
			block[i] = float32(rng.NormFloat64())
		}
		want := slices.Clone(block)
		rotatePairsGeneric(pairs, want)
		RotatePairs(pairs, block)
		for i := range block {
			if math.Float32bits(block[i]) != math.Float32bits(want[i]) {
				t.Fatalf("dims=%d index %d: kernel %v, generic %v", dims, i, block[i], want[i])
			}
		}
	}
}
