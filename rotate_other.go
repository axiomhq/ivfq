//go:build !amd64

package ivfq

func rotatePairs(pairs []pair, block []float32) { rotatePairsGeneric(pairs, block) }
