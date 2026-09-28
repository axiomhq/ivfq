package simd

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"
)

// The dispatched kernels are bitProductGeneric, count for count, at every
// width's tail, for a decoded row and for a row's bytes at every alignment
// a borrowed column can put it at.
func TestBitProductMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 1))
	for words := 0; words <= 20; words++ {
		for trial := 0; trial < 20; trial++ {
			row := make([]uint64, words)
			planes := make([]uint64, 4*words)
			fill := func(s []uint64) {
				for i := range s {
					switch trial % 4 {
					case 0:
						s[i] = rng.Uint64()
					case 1:
						s[i] = ^uint64(0)
					case 2:
						s[i] = rng.Uint64() & rng.Uint64() & rng.Uint64()
					}
				}
			}
			fill(row)
			fill(planes)
			wantOnes, wantWeighted := bitProductGeneric(row, planes)
			rowBytes := make([]byte, 8*words)
			for j, w := range row {
				binary.LittleEndian.PutUint64(rowBytes[8*j:], w)
			}
			if ones, weighted := bitProductGenericBytes(rowBytes, planes, words); ones != wantOnes || weighted != wantWeighted {
				t.Fatalf("words=%d trial=%d: generic bytes %d/%d, generic %d/%d", words, trial, ones, weighted, wantOnes, wantWeighted)
			}
			if ones, weighted := BitProductWords(row, planes); ones != wantOnes || weighted != wantWeighted {
				t.Fatalf("words=%d trial=%d: words kernel %d/%d, generic %d/%d", words, trial, ones, weighted, wantOnes, wantWeighted)
			}
			for shift := 0; shift < 8; shift++ {
				buf := make([]byte, shift+8*words)
				for j, w := range row {
					binary.LittleEndian.PutUint64(buf[shift+8*j:], w)
				}
				if ones, weighted := BitProductBytes(buf[shift:], planes, words); ones != wantOnes || weighted != wantWeighted {
					t.Fatalf("words=%d trial=%d shift=%d: bytes kernel %d/%d, generic %d/%d", words, trial, shift, ones, weighted, wantOnes, wantWeighted)
				}
			}
		}
	}
}
