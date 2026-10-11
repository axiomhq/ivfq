package simd

import "golang.org/x/sys/cpu"

var hasFWHT = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// fwht8 transforms each aligned run of 8 floats of x[:n] (n a multiple of 8)
// by the 8-point Hadamard matrix, in registers.
//
//go:noescape
func fwht8(x *float32, n int)

// fwhtStages runs the butterfly stages h = h0, 2*h0, ... < n over x[:n]:
// runs of h floats are added to and subtracted from the next run. h0 and n
// are multiples of 8 and n/h0 is a power of two.
//
//go:noescape
func fwhtStages(x *float32, n, h0 int)

// permScale is PermScale for n a multiple of 4.
//
//go:noescape
func permScale(dst, src *float32, perm *int32, coef *float32, n int)

// permScaleRows is PermScaleRows for n rows.
//
//go:noescape
func permScaleRows(dst, src *float32, perm *int32, coef *float32, n int)

// scale is Scale for n a multiple of 8.
//
//go:noescape
func scale(dst, src, coef *float32, n int)
