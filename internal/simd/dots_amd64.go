package simd

import (
	"golang.org/x/sys/cpu"
)

var hasAVX2FMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// Dots sets out[r] to Dot(q, block[r*len(q):(r+1)*len(q)]) for
// every row of block, bit for bit — the same kernel sequence go-simd runs
// per row — in one call: a centroid tree ranks a node's hundred children
// at a time, and at 16 dimensions the call per child cost more than the
// arithmetic.
func Dots(q, block, out []float32) {
	dims := len(q)
	if dims == 0 || len(block) != dims*len(out) {
		dotsGeneric(q, block, out)
		return
	}
	if hasAVX2FMA && len(out) > 0 {
		dotsAVX2FMA(&q[0], &block[0], dims, len(out), &out[0])
		return
	}
	dotsGeneric(q, block, out)
}

//go:noescape
func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32)
