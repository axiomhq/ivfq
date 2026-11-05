package simd

// Dots sets out[r] to Dot(q, block[r*len(q):(r+1)*len(q)]) for every row of
// block, bit for bit, in one call: a centroid tree ranks a node's hundred
// children at a time, and at 16 dimensions the call per child cost more
// than the arithmetic.
func Dots(q, block, out []float32) {
	dims := len(q)
	switch {
	case dims == 0 || len(out) == 0 || len(block) != dims*len(out):
		dotsGeneric(q, block, out)
	case hasAVX512F:
		dotsF32AVX512(&q[0], &block[0], dims, len(out), &out[0])
	case hasAVX2FMA:
		dotsAVX2FMA(&q[0], &block[0], dims, len(out), &out[0])
	default:
		dotsGeneric(q, block, out)
	}
}

// dotsGeneric is Dots in Go: Dot per row.
func dotsGeneric(q, block, out []float32) {
	dims := len(q)
	for r := range out {
		out[r] = Dot(q, block[r*dims:(r+1)*dims])
	}
}
