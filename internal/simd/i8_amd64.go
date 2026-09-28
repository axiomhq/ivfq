package simd

import "golang.org/x/sys/cpu"

// DecodeI8 is DecodeI8Into's loop; eight lanes at a time with AVX2.
func DecodeI8(dst []float32, b []byte) {
	i := 0
	if cpu.X86.HasAVX2 && len(dst) >= 8 {
		i = len(dst) &^ 7
		decodeI8AVX2(&dst[0], &b[0], i)
	}
	for ; i < len(dst); i++ {
		dst[i] = float32(int8(b[i]))
	}
}

// AppendI8 is AppendI8's loop. The vector kernel takes the leading run of
// eight-lane groups that are all integers — every row of an i8 column,
// which a rewrite re-encodes — where rounding is the identity and
// clamping is int8 saturation; RoundI8 takes the rest.
func AppendI8(dst []byte, v []float32) []byte {
	at := len(dst)
	dst = append(dst, make([]byte, len(v))...)
	out := dst[at:]
	i := 0
	if cpu.X86.HasAVX2 && len(v) >= 8 {
		i = encodeI8AVX2(&out[0], &v[0], len(v)&^7)
	}
	for ; i < len(v); i++ {
		out[i] = RoundI8(v[i])
	}
	return dst
}

//go:noescape
func decodeI8AVX2(dst *float32, src *byte, n int)

//go:noescape
func encodeI8AVX2(dst *byte, src *float32, n int) int
