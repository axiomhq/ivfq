package simd

// dotsGeneric is Dots in Go: Dot per row.
func dotsGeneric(q, block, out []float32) {
	dims := len(q)
	for r := range out {
		out[r] = Dot(q, block[r*dims:(r+1)*dims])
	}
}
