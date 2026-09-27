package ivfq

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
			if ones, weighted := bitProductWords(row, planes); ones != wantOnes || weighted != wantWeighted {
				t.Fatalf("words=%d trial=%d: words kernel %d/%d, generic %d/%d", words, trial, ones, weighted, wantOnes, wantWeighted)
			}
			for shift := 0; shift < 8; shift++ {
				buf := make([]byte, shift+8*words)
				for j, w := range row {
					binary.LittleEndian.PutUint64(buf[shift+8*j:], w)
				}
				if ones, weighted := bitProductBytes(buf[shift:], planes, words); ones != wantOnes || weighted != wantWeighted {
					t.Fatalf("words=%d trial=%d shift=%d: bytes kernel %d/%d, generic %d/%d", words, trial, shift, ones, weighted, wantOnes, wantWeighted)
				}
			}
		}
	}
}

// A scorer rebound to another column in the same frame (Scorer.With, the
// tier's per-cluster binding) scores that column's rows exactly as a
// scorer bound to it directly, whichever representation either column has
// and however many rows it holds.
func TestReboundScorerReadsTheNewColumn(t *testing.T) {
	for _, metric := range []string{"l2", "cosine"} {
		opts := Options{Dims: 129, Metric: metric, Seed: 11}
		centroid := corpus(1, 129)[0]
		small, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if small, err = AppendRows(small, corpus(5, 129)); err != nil {
			t.Fatal(err)
		}
		big, err := Empty(centroid, opts)
		if err != nil {
			t.Fatal(err)
		}
		if big, err = AppendRows(big, corpus(64, 129)[5:]); err != nil {
			t.Fatal(err)
		}
		data, err := big.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		bigBorrowed, err := UnmarshalBinaryBorrowed(data)
		if err != nil {
			t.Fatal(err)
		}
		q := NewQuery(corpus(2, 129)[1], metric)
		for _, target := range []Quantizer{big, bigBorrowed} {
			if !small.SameFrame(&target) {
				t.Fatalf("%s: columns are not in one frame", metric)
			}
			rebound := small.Scorer(q).With(&target)
			direct := target.Scorer(q)
			for row := 0; row < target.Rows(); row++ {
				gs, gb := rebound.ScoreAndBound(row)
				ws, wb := direct.ScoreAndBound(row)
				if gs != ws || gb != wb {
					t.Fatalf("%s row %d: rebound (%v,%v), direct (%v,%v)", metric, row, gs, gb, ws, wb)
				}
			}
		}
	}
}
