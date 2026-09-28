package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var hasBitProductAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasPOPCNT

func BitProductWords(row, planes []uint64) (ones, weighted uint64) {
	if hasBitProductAVX2 && len(row) > 0 {
		return bitProductAVX2((*byte)(unsafe.Pointer(&row[0])), &planes[0], len(row))
	}
	return bitProductGeneric(row, planes)
}

// BitProductBytes reads the row straight out of the column: amd64 is
// little-endian and the kernel loads unaligned.
func BitProductBytes(row []byte, planes []uint64, words int) (ones, weighted uint64) {
	if hasBitProductAVX2 && words > 0 {
		_ = row[:8*words]
		return bitProductAVX2(&row[0], &planes[0], words)
	}
	return bitProductGenericBytes(row, planes, words)
}

// bitProductAVX2 is bitProductGeneric over words words at row and the four
// planes at planes, planes+words, ... : 32 bytes at a time with the
// nibble-table popcount (VPSHUFB + VPSADBW), the last words%4 with POPCNTQ.
//
//go:noescape
func bitProductAVX2(row *byte, planes *uint64, words int) (ones, weighted uint64)
