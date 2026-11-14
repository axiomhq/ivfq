package simd

// Limit, for tests in other packages, turns off the AVX-512 kernels (avx512
// false) or every vector kernel (avx2 false), leaving the Go reference, and
// returns what restores them. A level the CPU lacks stays off.
func Limit(avx2, avx512 bool) (restore func()) {
	old := [5]bool{hasAVX2, hasAVX2POPCNT, hasAVX2FMA, hasAVX512F, hasVNNI}
	hasAVX2, hasAVX2POPCNT, hasAVX2FMA = old[0] && avx2, old[1] && avx2, old[2] && avx2
	hasAVX512F, hasVNNI = old[3] && avx2 && avx512, old[4] && avx2 && avx512
	return func() {
		hasAVX2, hasAVX2POPCNT, hasAVX2FMA, hasAVX512F, hasVNNI = old[0], old[1], old[2], old[3], old[4]
	}
}

// Has reports whether the AVX2 and the AVX-512 kernels run on this CPU.
func Has() (avx2, avx512 bool) { return hasAVX2, hasAVX512F }
