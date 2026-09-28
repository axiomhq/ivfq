//go:build !amd64

package simd

func DecodeI8(dst []float32, b []byte) {
	for i := range dst {
		dst[i] = float32(int8(b[i]))
	}
}

func AppendI8(dst []byte, v []float32) []byte {
	for _, x := range v {
		dst = append(dst, RoundI8(x))
	}
	return dst
}
