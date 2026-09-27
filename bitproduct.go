package ivfq

import (
	"encoding/binary"
	"math/bits"
)

// The 1-bit scan's inner product: for one row of sign bits and the query's
// four bit planes (rowIP), ones is popcount(row) and weighted is
// Σ_p 2^p popcount(row & plane_p). bitProductWords takes a decoded row,
// bitProductBytes the row's bytes inside a borrowed DQB1 column; both
// dispatch to the AVX2 kernel on amd64 and to bitProductGeneric elsewhere.
// planes holds the four bit planes back to back, words words each.

// bitProductGeneric is the reference: five POPCNTs per word.
func bitProductGeneric(row, planes []uint64) (ones, weighted uint64) {
	words := len(row)
	p1, p2, p3, p4 := planes[:words], planes[words:2*words], planes[2*words:3*words], planes[3*words:4*words]
	var c0, c1, c2, c3, c4 int
	for j, w := range row {
		c0 += bits.OnesCount64(w)
		c1 += bits.OnesCount64(w & p1[j])
		c2 += bits.OnesCount64(w & p2[j])
		c3 += bits.OnesCount64(w & p3[j])
		c4 += bits.OnesCount64(w & p4[j])
	}
	return uint64(c0), uint64(c1 + 2*c2 + 4*c3 + 8*c4)
}

// bitProductGenericBytes is bitProductGeneric over a row's little-endian
// bytes, words words long.
func bitProductGenericBytes(row []byte, planes []uint64, words int) (ones, weighted uint64) {
	_ = row[:8*words]
	p1, p2, p3, p4 := planes[:words], planes[words:2*words], planes[2*words:3*words], planes[3*words:4*words]
	var c0, c1, c2, c3, c4 int
	for j := 0; j < words; j++ {
		w := binary.LittleEndian.Uint64(row[8*j:])
		c0 += bits.OnesCount64(w)
		c1 += bits.OnesCount64(w & p1[j])
		c2 += bits.OnesCount64(w & p2[j])
		c3 += bits.OnesCount64(w & p3[j])
		c4 += bits.OnesCount64(w & p4[j])
	}
	return uint64(c0), uint64(c1 + 2*c2 + 4*c3 + 8*c4)
}
