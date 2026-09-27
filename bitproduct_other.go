//go:build !amd64

package ivfq

func bitProductWords(row, planes []uint64) (ones, weighted uint64) {
	return bitProductGeneric(row, planes)
}

func bitProductBytes(row []byte, planes []uint64, words int) (ones, weighted uint64) {
	return bitProductGenericBytes(row, planes, words)
}
