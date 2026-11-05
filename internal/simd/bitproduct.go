package simd

import (
	"encoding/binary"
	"math/bits"
	"unsafe"
)

// BitProductWords is the 1-bit scan's inner product of a decoded row of
// sign bits and the query's four bit planes (rowIP): ones is popcount(row)
// and weighted is Σ_p 2^p popcount(row & plane_p). planes holds the four
// planes back to back, len(row) words each.
func BitProductWords(row, planes []uint64) (ones, weighted uint64) {
	if hasAVX2POPCNT && len(row) > 0 {
		return bitProductAVX2((*byte)(unsafe.Pointer(&row[0])), &planes[0], len(row))
	}
	return bitProductGeneric(row, planes)
}

// BitProductBytes is BitProductWords over the little-endian bytes of the
// row's words inside a borrowed DQB1 column. The kernel reads them in place:
// amd64 is little-endian and the kernel loads unaligned.
func BitProductBytes(row []byte, planes []uint64, words int) (ones, weighted uint64) {
	if hasAVX2POPCNT && words > 0 {
		_ = row[:8*words]
		return bitProductAVX2(&row[0], &planes[0], words)
	}
	return bitProductGenericBytes(row, planes, words)
}

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
