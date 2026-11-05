package simd

// fastScanAVX2 is fastScanGeneric with pairs = groups/2: per block, two
// groups per 32-byte load, each nibble half looked up with VPSHUFB, the
// byte results split into even and odd 16-bit lanes and summed; the two
// 128-bit lanes (one group each) are folded and interleaved back into row
// order at the end of the block.
//
//go:noescape
func fastScanAVX2(nibs *byte, lut *byte, pairs, blocks int, out *uint16)

// fastScan2AVX2 is fastScanAVX2 with a second table, the nibbles split once.
//
//go:noescape
func fastScan2AVX2(nibs, lutA, lutB *byte, pairs, blocks int, outA, outB *uint16)
