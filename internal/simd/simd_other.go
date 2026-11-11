//go:build !amd64

package simd

// No vector kernels here: every flag is false, so no stub below is called.
// The flags are variables because the tests clear them on amd64.
var hasAVX2, hasAVX2POPCNT, hasAVX2FMA, hasAVX512F, hasVNNI bool

func bitProductAVX2(row *byte, planes *uint64, words int) (ones, weighted uint64) {
	panic("unreachable")
}

func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32) { panic("unreachable") }
func dotsF32AVX512(q *float32, block *float32, dims, rows int, out *float32)   { panic("unreachable") }
func l2sqF32AVX512(a *float32, b *float32, n int) float32                      { panic("unreachable") }

func dotsI8AVX2(q *int8, block *int8, dims, wide, rows int, out *int32) { panic("unreachable") }
func dotsI8VNNI(q *int8, block *int8, dims, rows int, out *int32)       { panic("unreachable") }

func fastScanAVX2(nibs *byte, lut *byte, pairs, blocks int, out *uint16) { panic("unreachable") }
func fastScan2AVX2(nibs, lutA, lutB *byte, pairs, blocks int, outA, outB *uint16) {
	panic("unreachable")
}

func fwht8(x *float32, n int)                                            { panic("unreachable") }
func fwhtStages(x *float32, n, h0 int)                                   { panic("unreachable") }
func permScale(dst, src *float32, perm *int32, coef *float32, n int)     { panic("unreachable") }
func permScaleRows(dst, src *float32, perm *int32, coef *float32, n int) { panic("unreachable") }
func scale(dst, src, coef *float32, n int)                               { panic("unreachable") }

func decodeI8AVX2(dst *float32, src *byte, n int)     { panic("unreachable") }
func encodeI8AVX2(dst *byte, src *float32, n int) int { panic("unreachable") }

func blockSumSq(block *float32, dims int, sums *float64)         { panic("unreachable") }
func blockBits(block *float32, n int, abs *float64, out *uint32) { panic("unreachable") }
func minMaxScaled(x *float32, n int, mul float64) (lo, hi float64) {
	panic("unreachable")
}
func sumSq(x *float32, n int) float64 { panic("unreachable") }
func quantCodes(codes *byte, x *float32, n int, mul, lo, delta, levels float64) float64 {
	panic("unreachable")
}

func nonFinite(x *float32, n int) bool         { panic("unreachable") }
func addF64(dst *float64, src *float32, n int) { panic("unreachable") }
func scaleF64(x *float32, n int, s float64)    { panic("unreachable") }
