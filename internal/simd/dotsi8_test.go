package simd

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// DotsI8 is the integer dot product, exactly, at every width's tail and at
// the extremes of int8 (-128 · -128 in every lane).
func TestDotsI8MatchesGeneric(t *testing.T) {
	// Every kernel the CPU has: the AVX-512 VNNI one, then AVX2 alone.
	for _, vnni := range []bool{true, false} {
		if vnni && !hasDotsI8VNNI {
			continue
		}
		old := hasDotsI8VNNI
		hasDotsI8VNNI = vnni
		testDotsI8(t)
		hasDotsI8VNNI = old
	}
}

func testDotsI8(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, dims := range []int{1, 15, 16, 17, 31, 32, 63, 64, 65, 100, 127, 128, 129, 191, 192, 193, 768, 769} {
		for _, rows := range []int{0, 1, 3, 7} {
			q := make([]int8, dims)
			block := make([]int8, dims*rows)
			for i := range q {
				q[i] = int8(r.IntN(256) - 128)
			}
			for i := range block {
				block[i] = int8(r.IntN(256) - 128)
			}
			if dims == 768 || dims == 193 {
				for i := range q {
					q[i] = -128
				}
				for i := range block {
					block[i] = -128
				}
			}
			got, want := make([]int32, rows), make([]int32, rows)
			DotsI8(q, block, got)
			dotsI8Generic(q, block, want)
			if !slices.Equal(got, want) {
				t.Fatalf("dims %d rows %d: %v, want %v", dims, rows, got, want)
			}
		}
	}
}

func BenchmarkDotsI8(b *testing.B) {
	q := make([]int8, 768)
	block := make([]int8, 768*1600)
	out := make([]int32, 1600)
	for b.Loop() {
		DotsI8(q, block, out)
	}
}
