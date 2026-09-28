//go:build !amd64

package simd

func BitProductWords(row, planes []uint64) (ones, weighted uint64) {
	return bitProductGeneric(row, planes)
}

func BitProductBytes(row []byte, planes []uint64, words int) (ones, weighted uint64) {
	return bitProductGenericBytes(row, planes, words)
}
