//go:build !amd64

package simd

func RotatePairs(pairs []Pair, block []float32) { rotatePairsGeneric(pairs, block) }
