package simd

// DotsI8 sets out[r] to the integer dot product of q and row r of block
// (len(q) wide), exactly: the int8 centroid scan's kernel. Products and
// sums fit int32 up to 133,000 dimensions.
func DotsI8(q, block []int8, out []int32) {
	dims := len(q)
	if dims == 0 || len(block) != dims*len(out) {
		dotsI8Generic(q, block, out)
		return
	}
	dotsI8(q, block, out)
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
