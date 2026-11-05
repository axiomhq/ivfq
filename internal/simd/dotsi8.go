package simd

// DotsI8 sets out[r] to the integer dot product of q and row r of block
// (len(q) wide), exactly: the int8 centroid scan's kernel. Products and
// sums fit int32 below 131,072 dimensions.
func DotsI8(q, block []int8, out []int32) {
	dims := len(q)
	if dims == 0 || len(out) == 0 || len(block) != dims*len(out) || !hasAVX2 {
		dotsI8Generic(q, block, out)
		return
	}
	if hasVNNI {
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

func dotsI8Generic(q, block []int8, out []int32) {
	dims := len(q)
	for r := range out {
		var s int32
		for j, x := range block[r*dims : (r+1)*dims] {
			s += int32(x) * int32(q[j])
		}
		out[r] = s
	}
}
