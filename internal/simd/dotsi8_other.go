//go:build !amd64

package simd

func dotsI8(q, block []int8, out []int32) { dotsI8Generic(q, block, out) }

// No vector kernel here; the tests toggle this on amd64.
var hasDotsI8VNNI bool
