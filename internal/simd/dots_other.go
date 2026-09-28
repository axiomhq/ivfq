//go:build !amd64

package simd

// Dots sets out[r] to Dot(q, block[r*len(q):(r+1)*len(q)]) for
// every row of block.
func Dots(q, block, out []float32) { dotsGeneric(q, block, out) }
