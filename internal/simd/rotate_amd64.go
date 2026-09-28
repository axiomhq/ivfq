package simd

import "golang.org/x/sys/cpu"

// RotatePairs applies each pair's Givens rotation to a block of FillBlock
// rows: eight lanes at a time with AVX2, in Go without it.
func RotatePairs(pairs []Pair, block []float32) {
	if cpu.X86.HasAVX2 {
		rotatePairsAVX2(pairs, &block[0])
		return
	}
	rotatePairsGeneric(pairs, block)
}

// rotatePairsAVX2 is rotatePairsGeneric for FillBlock == 16: two YMM
// registers per dimension, multiplies and adds rounded separately (no
// FMA), so the result is bit-identical. Every pair's indexes must address
// block; the caller only makes pairs inside its dims and checks
// the block's length.
//
//go:noescape
func rotatePairsAVX2(pairs []Pair, block *float32)
