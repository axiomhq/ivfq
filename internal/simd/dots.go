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

// Dots3 sets out[3*r+k] to the dot of row k of (a, b, c) with row r of
// block, len(a) wide: three rows against a block in one pass, each block
// load shared by the three. Twelve independent sums at a time (three rows
// by four block rows) keep the FMA units busy where Dots, one sum per row,
// waits on each FMA's latency. Each pair's sum is the same whichever rows
// share its call, but it is not Dot's: it sums eight lanes then folds them.
// a, b and c must be as long; block must hold len(out)/3 rows.
func Dots3(a, b, c, block, out []float32) {
	dims, rows := len(a), len(out)/3
	if len(b) != dims || len(c) != dims || len(out) != 3*rows || len(block) != dims*rows {
		panic("simd: Dots3 sizes disagree")
	}
	if rows == 0 {
		return
	}
	if hasAVX2FMA && dims >= 8 && dims%8 == 0 {
		dots3AVX2FMA(&a[0], &b[0], &c[0], &block[0], dims, rows, &out[0])
		return
	}
	for r := range rows {
		row := block[r*dims : (r+1)*dims]
		out[3*r], out[3*r+1], out[3*r+2] = Dot(a, row), Dot(b, row), Dot(c, row)
	}
}
