package ivfq

import "golang.org/x/sys/cpu"

// rotatePairs applies each pair's Givens rotation to a block of fillBlock
// rows: eight lanes at a time with AVX2, in Go without it.
func rotatePairs(pairs []pair, block []float32) {
	if cpu.X86.HasAVX2 {
		rotatePairsAVX2(pairs, &block[0])
		return
	}
	rotatePairsGeneric(pairs, block)
}

// rotatePairsAVX2 is rotatePairsGeneric for fillBlock == 16: two YMM
// registers per dimension, multiplies and adds rounded separately (no
// FMA), so the result is bit-identical. Every pair's indexes must address
// block; newRotation only makes pairs inside its dims and applyBlock checks
// the block's length.
//
//go:noescape
func rotatePairsAVX2(pairs []pair, block *float32)
