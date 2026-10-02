//go:build !amd64

package simd

func fastScan(nibs, lut []byte, groups, blocks int, out []uint16) {
	fastScanGeneric(nibs, lut, groups, blocks, out)
}
