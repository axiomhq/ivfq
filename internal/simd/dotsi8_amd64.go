package simd

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
