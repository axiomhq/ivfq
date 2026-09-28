package simd

import (
	"encoding/binary"
	"fmt"
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

// BenchmarkBitProduct is one row's five popcounts: the dispatched kernel

// against the generic loop, at the widths that matter.

func BenchmarkBitProduct(b *testing.B) {
	for _, dims := range []int{128, 768, 1536} {
		words := (dims + 63) / 64
		row := make([]uint64, words)
		planes := make([]uint64, 4*words)
		for i := range row {
			row[i] = uint64(i)*0x9E3779B97F4A7C15 ^ 0x5555
		}
		for i := range planes {
			planes[i] = uint64(i)*0xC2B2AE3D27D4EB4F ^ 0x3333
		}
		b.Run(fmt.Sprintf("generic/%d", dims), func(b *testing.B) {
			var s uint64
			for b.Loop() {
				o, w := bitProductGeneric(row, planes)
				s += o + w
			}
			_ = s
		})
		b.Run(fmt.Sprintf("kernel/%d", dims), func(b *testing.B) {
			var s uint64
			for b.Loop() {
				o, w := BitProductWords(row, planes)
				s += o + w
			}
			_ = s
		})
	}
}
