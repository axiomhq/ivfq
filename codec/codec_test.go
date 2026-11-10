package codec

import (
	"encoding/binary"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/x448/float16"
)

// TestF16RoundTripAndOverflow pins what f16 validation relies on: every
// f16-representable value survives the codec, and the first float32 that
// rounds past f16's max (65520; 65519 still rounds down to 65504) decodes as
// Inf so validation can reject it. The boundary rows are what a consumer
// observes at f16's edges: the sign of zero, the smallest subnormal, the
// underflow to signed zero below half of it, and NaN staying NaN.
func TestF16RoundTripAndOverflow(t *testing.T) {
	exact := []float32{0, 1, -1.5, 0.5, 65504, -65504, 5.9604645e-08, -5.9604645e-08, 1.0009766}
	if got := DecodeF16(EncodeF16(exact)); !reflect.DeepEqual(got, exact) {
		t.Fatalf("round trip %v -> %v", exact, got)
	}
	got := DecodeF16(EncodeF16([]float32{65519, 65520, -65520, math.MaxFloat32}))
	if got[0] != 65504 || !math.IsInf(float64(got[1]), 1) || !math.IsInf(float64(got[2]), -1) || !math.IsInf(float64(got[3]), 1) {
		t.Fatalf("overflow edge: %v", got)
	}
	negZero := float32(math.Copysign(0, -1))
	belowHalfMin := float32(2.9e-08) // under half the smallest subnormal: rounds to zero
	got = DecodeF16(EncodeF16([]float32{negZero, belowHalfMin, -belowHalfMin, float32(math.NaN())}))
	if got[0] != 0 || !math.Signbit(float64(got[0])) {
		t.Fatalf("-0 -> %v (signbit %v)", got[0], math.Signbit(float64(got[0])))
	}
	if got[1] != 0 || math.Signbit(float64(got[1])) || got[2] != 0 || !math.Signbit(float64(got[2])) {
		t.Fatalf("underflow below half the smallest subnormal: %v, %v", got[1], got[2])
	}
	if !math.IsNaN(float64(got[3])) {
		t.Fatalf("NaN -> %v", got[3])
	}
}

// The stored bytes are the wire format: little-endian binary16, and int8
// rounded half away from zero and saturated.
func TestStorageBytesAndI8Saturation(t *testing.T) {
	v := []float32{-200, -1.5, 0, 1.5, 200}
	if got, want := EncodeF16(v), []byte{0x40, 0xda, 0x00, 0xbe, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x5a}; !slices.Equal(got, want) {
		t.Fatalf("EncodeF16(%v) = % x, want % x", v, got, want)
	}
	i := EncodeI8(v)
	if want := []byte{0x80, 0xfe, 0x00, 0x02, 0x7f}; !slices.Equal(i, want) {
		t.Fatalf("EncodeI8(%v) = % x, want % x", v, i, want)
	}
	d := DecodeI8(i)
	want := []float32{-128, -2, 0, 2, 127}
	for j := range d {
		if d[j] != want[j] {
			t.Fatal(d)
		}
	}
}

func TestRoundStoredKeepsExactRowsWithoutAllocating(t *testing.T) {
	exact := []float32{-128, -1, 0, 1, 127}
	if got := RoundI8(exact); &got[0] != &exact[0] {
		t.Error("RoundI8 copied a row already at i8 precision")
	}
	if got := RoundF16(exact); &got[0] != &exact[0] {
		t.Error("RoundF16 copied a row already at f16 precision")
	}
	if n := testing.AllocsPerRun(100, func() { RoundI8(exact); RoundF16(exact) }); n != 0 {
		t.Errorf("%v allocations for exact rows", n)
	}
	for _, v := range [][]float32{{0.5, -0.5, 2.5, 127.6, -128.4, 300, -300}, {float32(math.Copysign(0, -1))}, {1e-3, 65519}, {128, -129}} {
		if got, want := RoundI8(v), DecodeI8(EncodeI8(v)); !slices.Equal(got, want) || math.Signbit(float64(got[0])) != math.Signbit(float64(want[0])) {
			t.Errorf("RoundI8(%v) = %v, want %v", v, got, want)
		}
		if got, want := RoundF16(v), DecodeF16(EncodeF16(v)); !slices.Equal(got, want) {
			t.Errorf("RoundF16(%v) = %v, want %v", v, got, want)
		}
	}
}

// The vector i8 codecs are the scalar ones: integral groups (in and out
// of range, -0, huge) through the kernel, groups with a fraction or a NaN
// through roundI8, and every length's remainder.
func TestI8CodecsMatchScalar(t *testing.T) {
	special := []float32{0, float32(math.Copysign(0, -1)), 1, -1, 127, -128, 128, -129, 300, -300, 1e9, -1e9,
		2147483648, -2147483904, 0.5, -0.5, 2.5, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	rng := uint32(1)
	next := func() uint32 { rng = rng*1664525 + 1013904223; return rng }
	for n := range 70 {
		for trial := range 20 {
			v := make([]float32, n)
			for i := range v {
				switch {
				case trial%3 == 0:
					v[i] = float32(int8(next()))
				case trial%3 == 1:
					v[i] = special[next()%uint32(len(special))]
				default:
					v[i] = float32(int32(next()%600) - 300)
				}
			}
			want := make([]byte, n)
			for i, x := range v {
				want[i] = scalarI8(x)
			}
			if got := AppendI8([]byte{9}, v); !slices.Equal(got[1:], want) || got[0] != 9 {
				t.Fatalf("AppendI8(%v) = %v, want %v", v, got[1:], want)
			}
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(next())
			}
			dec := make([]float32, n)
			DecodeI8Into(dec, b)
			for i := range b {
				if dec[i] != float32(int8(b[i])) {
					t.Fatalf("DecodeI8Into(%v)[%d] = %v", b, i, dec[i])
				}
			}
		}
	}
}

// A short b panics rather than decoding bytes past its length, even when
// its spare capacity holds them.
func TestDecodeShortPanics(t *testing.T) {
	for _, n := range []int{5, 16} {
		b := make([]byte, n/2, 2*n)
		for name, f := range map[string]func(){
			"DecodeI8Into":  func() { DecodeI8Into(make([]float32, n), b) },
			"DecodeF16Into": func() { DecodeF16Into(make([]float32, n), b) },
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("%s of %d values from %d bytes did not panic", name, n, len(b))
					}
				}()
				f()
			}()
		}
	}
}

// scalarI8 is EncodeI8's definition.
func scalarI8(x float32) byte {
	if x != x {
		return 0
	}
	return byte(int8(max(-128, min(127, math.Round(float64(x))))))
}

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

func BenchmarkDecodeF16(b *testing.B) {
	v := make([]float32, 4096)
	for i := range v {
		v[i] = float32(i%97) * 0.37
	}
	enc := EncodeF16(v)
	dst := make([]float32, len(v))
	b.SetBytes(int64(len(enc)))
	for b.Loop() {
		DecodeF16Into(dst, enc)
	}
}

// DecodeF16 must match the library on every bit pattern, including zeros,
// subnormals, Inf and NaN.
func TestDecodeF16EveryHalf(t *testing.T) {
	b := make([]byte, 2<<16)
	for i := range 1 << 16 {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(i))
	}
	for i, got := range DecodeF16(b) {
		if want := float16.Frombits(uint16(i)).Float32(); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("half %#04x: got %#08x, want %#08x", i, math.Float32bits(got), math.Float32bits(want))
		}
	}
}
