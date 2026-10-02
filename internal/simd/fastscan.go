package simd

// The batched 1-bit scan ("fast scan"): a column's rows are packed in
// blocks of 32, a nibble (four dimensions) of each row per group, and the
// query is a table per group of what each nibble contributes. One table
// lookup (VPSHUFB) then scores a group of 32 rows at once.
//
// Layout of nibs: block k, group g is 16 bytes at (k*groups+g)*16; byte i
// holds row 32k+i's nibble in its low half and row 32k+16+i's in its high
// half. lut holds groups tables of 16 bytes, table g at 16g. groups is
// even, so the AVX2 kernel reads two groups (32 bytes) at a time. Table
// entries must stay below 64 so a row's sum fits 16 bits.

// FastScan sets out[32k+i] to Σ_g lut[16g + nibble(row 32k+i, group g)]
// for every block k: len(nibs) = blocks*groups*16, len(lut) = groups*16,
// len(out) = blocks*32, groups even and at most FastScanMaxGroups.
func FastScan(nibs, lut []byte, groups, blocks int, out []uint16) {
	if groups <= 0 || groups%2 != 0 || groups > FastScanMaxGroups ||
		len(nibs) != blocks*groups*16 || len(lut) != groups*16 || len(out) != blocks*32 {
		panic("simd: FastScan shape")
	}
	if blocks == 0 {
		return
	}
	fastScan(nibs, lut, groups, blocks, out)
}

// FastScanMaxGroups bounds groups so a block's sums fit 16 bits with
// table entries below 64: 1,024 groups is 4,096 dimensions.
const FastScanMaxGroups = 1024

// fastScanGeneric is the reference.
func fastScanGeneric(nibs, lut []byte, groups, blocks int, out []uint16) {
	for k := range blocks {
		for i := range 32 {
			byteIdx, shift := i, uint(0)
			if i >= 16 {
				byteIdx, shift = i-16, 4
			}
			var sum uint16
			for g := range groups {
				n := nibs[(k*groups+g)*16+byteIdx] >> shift & 0xF
				sum += uint16(lut[16*g+int(n)])
			}
			out[32*k+i] = sum
		}
	}
}
