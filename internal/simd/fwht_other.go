//go:build !amd64

package simd

// No vector kernels here; the tests toggle this on amd64.
var hasFWHT bool

func fwht8(x *float32, n int)                                            { panic("unreachable") }
func fwhtStages(x *float32, n, h0 int)                                   { panic("unreachable") }
func permScale(dst, src *float32, perm *int32, coef *float32, n int)     { panic("unreachable") }
func permScaleRows(dst, src *float32, perm *int32, coef *float32, n int) { panic("unreachable") }
func scale(dst, src, coef *float32, n int)                               { panic("unreachable") }
