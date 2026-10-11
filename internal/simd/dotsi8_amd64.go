package simd

import "golang.org/x/sys/cpu"

var hasDotsI8AVX2 = cpu.X86.HasAVX2

// VNNI needs AVX-512 BW for the byte masks and F for the 512-bit registers.
var hasDotsI8VNNI = cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasAVX512VNNI

func dotsI8(q, block []int8, out []int32) {
	dims := len(q)
	if !hasDotsI8AVX2 || len(out) == 0 {
		dotsI8Generic(q, block, out)
		return
	}
	if hasDotsI8VNNI {
		dotsI8VNNI(&q[0], &block[0], dims, len(out), &out[0])
		return
	}
	wide := dims &^ 15
	if wide > 0 {
		dotsI8AVX2(&q[0], &block[0], dims, wide, len(out), &out[0])
	} else {
		clear(out)
	}
	if wide < dims { // the tail, a row at a time
		for r := range out {
			s := out[r]
			for j := wide; j < dims; j++ {
				s += int32(block[r*dims+j]) * int32(q[j])
			}
			out[r] = s
		}
	}
}

// dotsI8AVX2 sets out[r] to the dot product of the first wide (a multiple
// of 16) lanes of q and of row r (dims wide): sixteen lanes sign-extended
// to int16, multiplied and pair-summed to int32 (VPMADDWD) per step.
//
//go:noescape
func dotsI8AVX2(q *int8, block *int8, dims, wide, rows int, out *int32)

// dotsI8VNNI is dotsI8AVX2 for all dims (no tail), 64 lanes a step.
//
//go:noescape
func dotsI8VNNI(q *int8, block *int8, dims, rows int, out *int32)
