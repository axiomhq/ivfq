//go:build !amd64

package ivfq

func decodeI8(dst []float32, b []byte) {
	for i := range dst {
		dst[i] = float32(int8(b[i]))
	}
}

func appendI8(dst []byte, v []float32) []byte {
	for _, x := range v {
		dst = append(dst, roundI8(x))
	}
	return dst
}
