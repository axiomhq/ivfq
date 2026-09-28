package simd

// FillBlock is how many rows RotatePairs rotates together. The AVX2 kernel
// is written for exactly this width.
const FillBlock = 16

// Pair is one Givens rotation of a butterfly: coordinates i and j turned
// by the angle whose cosine and sine are cs and sn.
type Pair struct {
	I, J   int32
	Cs, Sn float32
}

// rotatePairsGeneric is RotatePairs in Go. The explicit float32
// conversions keep the compiler from fusing a multiply into the add on
// platforms that have FMA: each product and sum is rounded on its own, as
// the vector kernel rounds them.
func rotatePairsGeneric(pairs []Pair, block []float32) {
	for _, p := range pairs {
		vi := block[int(p.I)*FillBlock:][:FillBlock]
		vj := block[int(p.J)*FillBlock:][:FillBlock]
		for k, a := range vi {
			b := vj[k]
			vi[k], vj[k] = float32(p.Cs*a)-float32(p.Sn*b), float32(p.Sn*a)+float32(p.Cs*b)
		}
	}
}
