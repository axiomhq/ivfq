//go:build !amd64

package simd

// No vector kernels here; the tests toggle this on amd64.
var hasFWHT bool

func fwht8(x *float32, n int)                                            { panic("unreachable") }
func fwhtStages(x *float32, n, h0 int)                                   { panic("unreachable") }
func permScale(dst, src *float32, perm *int32, coef *float32, n int)     { panic("unreachable") }
func permScaleRows(dst, src *float32, perm *int32, coef *float32, n int) { panic("unreachable") }
func scale(dst, src, coef *float32, n int)                               { panic("unreachable") }

func blockSumSq(block *float32, dims int, sums *float64)         { panic("unreachable") }
func blockBits(block *float32, n int, abs *float64, out *uint32) { panic("unreachable") }
func minMaxScaled(x *float32, n int, mul float64) (lo, hi float64) {
	panic("unreachable")
}
func sumSq(x *float32, n int) float64 { panic("unreachable") }
func quantCodes(codes *byte, x *float32, n int, mul, lo, delta, levels float64) float64 {
	panic("unreachable")
}
