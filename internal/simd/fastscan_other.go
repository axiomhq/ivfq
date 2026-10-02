//go:build !amd64

package simd

func fastScan(nibs, lut []byte, groups, blocks int, out []uint16) {
	fastScanGeneric(nibs, lut, groups, blocks, out)
}

func fastScan2(nibs, lutA, lutB []byte, groups, blocks int, outA, outB []uint16) {
	fastScanGeneric(nibs, lutA, groups, blocks, outA)
	fastScanGeneric(nibs, lutB, groups, blocks, outB)
}
