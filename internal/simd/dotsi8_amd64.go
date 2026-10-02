package simd

import "golang.org/x/sys/cpu"

var hasDotsI8AVX2 = cpu.X86.HasAVX2

func dotsI8(q, block []int8, out []int32) {
	dims := len(q)
	if !hasDotsI8AVX2 || len(out) == 0 {
		dotsI8Generic(q, block, out)
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
