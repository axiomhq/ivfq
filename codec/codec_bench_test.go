package codec

import "testing"

// A hood of 1,000 i8 rows at 128 dims, as a rewrite decodes and
// re-encodes it.
func BenchmarkI8Column(b *testing.B) {
	const rows, dims = 1000, 128
	raw := make([]byte, rows*dims)
	for i := range raw {
		raw[i] = byte(i * 7)
	}
	vecs := make([]float32, len(raw))
	b.Run("decode", func(b *testing.B) {
		b.SetBytes(int64(len(raw)))
		for b.Loop() {
			DecodeI8Into(vecs, raw)
		}
	})
	out := make([]byte, 0, len(raw))
	b.Run("encode", func(b *testing.B) {
		b.SetBytes(int64(len(raw)))
		for b.Loop() {
			out = out[:0]
			for r := range rows {
				out = AppendI8(out, vecs[r*dims:(r+1)*dims])
			}
		}
	})
}
