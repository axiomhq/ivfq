// Package codec stores float32 rows at reduced precision: IEEE binary16
// (round to nearest even) and int8 (round half away from zero,
// saturating).
package codec

import (
	"encoding/binary"
	"github.com/axiomhq/ivfq/internal/simd"
	"math"

	"github.com/x448/float16"
)

// EncodeF16 stores v as little-endian IEEE binary16, round-to-nearest-even;
// out-of-range values become ±Inf (callers validate with
// float16.Fromfloat32(x).IsInf(0) before storing).
func EncodeF16(v []float32) []byte { return AppendF16(make([]byte, 0, 2*len(v)), v) }

// AppendF16 appends EncodeF16(v) to dst.
func AppendF16(dst []byte, v []float32) []byte {
	for _, x := range v {
		dst = binary.LittleEndian.AppendUint16(dst, float16.Fromfloat32(x).Bits())
	}
	return dst
}

func DecodeF16(b []byte) []float32 {
	v := make([]float32, len(b)/2)
	DecodeF16Into(v, b)
	return v
}

// DecodeF16Into decodes len(dst) values of b into dst.
func DecodeF16Into(dst []float32, b []byte) {
	_ = b[:2*len(dst)]
	for i := range dst {
		dst[i] = float16.Frombits(binary.LittleEndian.Uint16(b[2*i:])).Float32()
	}
}

// RoundF16 is DecodeF16(EncodeF16(v)), except that it returns v itself,
// allocating nothing, when every value already is an f16 value.
func RoundF16(v []float32) []float32 {
	for _, x := range v {
		if float16.Fromfloat32(x).Float32() != x {
			return DecodeF16(EncodeF16(v))
		}
	}
	return v
}

// EncodeI8 rounds to the nearest integer, ties away from zero, and saturates
// to [-128,127]. This is an implementation choice because the public docs do
// not prescribe conversion of floating-point inputs to i8 storage.
func EncodeI8(v []float32) []byte { return AppendI8(make([]byte, 0, len(v)), v) }

// AppendI8 appends EncodeI8(v) to dst.
func AppendI8(dst []byte, v []float32) []byte { return simd.AppendI8(dst, v) }

func DecodeI8(b []byte) []float32 {
	v := make([]float32, len(b))
	DecodeI8Into(v, b)
	return v
}

// DecodeI8Into decodes len(dst) values of b into dst.
func DecodeI8Into(dst []float32, b []byte) { simd.DecodeI8(dst, b[:len(dst)]) }

// RoundI8 is DecodeI8(EncodeI8(v)), except that it returns v itself,
// allocating nothing, when every value already is an i8 value — as every
// row read back from an i8 column is. -0 is not: it decodes as +0.
func RoundI8(v []float32) []float32 {
	for _, x := range v {
		if x < -128 || x > 127 || x != float32(int32(x)) || math.Float32bits(x) == 1<<31 {
			return DecodeI8(EncodeI8(v))
		}
	}
	return v
}
