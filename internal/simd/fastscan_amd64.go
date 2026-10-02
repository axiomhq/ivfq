package simd

import "golang.org/x/sys/cpu"

var hasFastScanAVX2 = cpu.X86.HasAVX2

func fastScan(nibs, lut []byte, groups, blocks int, out []uint16) {
	if hasFastScanAVX2 {
		fastScanAVX2(&nibs[0], &lut[0], groups/2, blocks, &out[0])
		return
	}
	fastScanGeneric(nibs, lut, groups, blocks, out)
}

// fastScanAVX2 is fastScanGeneric with pairs = groups/2: per block, two
// groups per 32-byte load, each nibble half looked up with VPSHUFB, the
// byte results split into even and odd 16-bit lanes and summed; the two
// 128-bit lanes (one group each) are folded and interleaved back into row
// order at the end of the block.
//
//go:noescape
func fastScanAVX2(nibs *byte, lut *byte, pairs, blocks int, out *uint16)

func fastScan2(nibs, lutA, lutB []byte, groups, blocks int, outA, outB []uint16) {
	if hasFastScanAVX2 {
		fastScan2AVX2(&nibs[0], &lutA[0], &lutB[0], groups/2, blocks, &outA[0], &outB[0])
		return
	}
	fastScanGeneric(nibs, lutA, groups, blocks, outA)
	fastScanGeneric(nibs, lutB, groups, blocks, outB)
}

// fastScan2AVX2 is fastScanAVX2 with a second table, the nibbles split once.
//
//go:noescape
func fastScan2AVX2(nibs, lutA, lutB *byte, pairs, blocks int, outA, outB *uint16)
