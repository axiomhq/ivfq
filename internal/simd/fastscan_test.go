package simd

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// FastScan is its reference on every shape, with tables at the largest
// entry the caller makes (60) so the 16-bit sums are pushed hard.
func TestFastScanMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 3))
	for _, groups := range []int{2, 4, 6, 32, 34, 200, FastScanMaxGroups} {
		for _, blocks := range []int{0, 1, 2, 5} {
			nibs := make([]byte, blocks*groups*16)
			for i := range nibs {
				nibs[i] = byte(rng.IntN(256))
			}
			lut := make([]byte, groups*16)
			for i := range lut {
				lut[i] = byte(rng.IntN(61))
			}
			if groups == FastScanMaxGroups {
				for i := range lut {
					lut[i] = 60
				}
			}
			got := make([]uint16, blocks*32)
			want := make([]uint16, blocks*32)
			FastScan(nibs, lut, groups, blocks, got)
			fastScanGeneric(nibs, lut, groups, blocks, want)
			if !slices.Equal(got, want) {
				t.Fatalf("groups=%d blocks=%d: kernel %v\nreference %v", groups, blocks, got, want)
			}
		}
	}
}

// The reference reads the layout FastScan documents: row 32k+i's nibble of
// group g is the low half of byte i for i < 16, the high half of byte i-16
// otherwise. One row and one group set, so a misplaced row or group shows.
func TestFastScanLayout(t *testing.T) {
	const groups, blocks = 4, 2
	for _, row := range []int{0, 5, 15, 16, 21, 31, 32, 47, 48, 63} {
		for g := range groups {
			nibs := make([]byte, blocks*groups*16)
			k, i := row/32, row%32
			if i < 16 {
				nibs[(k*groups+g)*16+i] = 0x3
			} else {
				nibs[(k*groups+g)*16+i-16] = 0x3 << 4
			}
			lut := make([]byte, groups*16)
			lut[16*g+3] = 7
			out := make([]uint16, blocks*32)
			FastScan(nibs, lut, groups, blocks, out)
			for r, v := range out {
				want := uint16(0)
				if r == row {
					want = 7
				}
				if v != want {
					t.Fatalf("row %d group %d: out[%d] = %d, want %d", row, g, r, v, want)
				}
			}
		}
	}
}

func BenchmarkFastScan(b *testing.B) {
	const groups, blocks = 32, 8 // 128 dims, 256 rows
	nibs := make([]byte, blocks*groups*16)
	lut := make([]byte, groups*16)
	for i := range nibs {
		nibs[i] = byte(i * 7)
	}
	for i := range lut {
		lut[i] = byte(i % 61)
	}
	out := make([]uint16, blocks*32)
	for b.Loop() {
		FastScan(nibs, lut, groups, blocks, out)
	}
}
