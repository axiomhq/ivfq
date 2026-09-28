package codec

import (
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"reflect"
	"slices"
	"testing"
)

// TestF16RoundTripAndOverflow pins what WF-04 validation relies on: every
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
func TestStorageSizesAndScores(t *testing.T) {
	v := []float32{-200, -1.5, 0, 1.5, 200}
	f := EncodeF16(v)
	i := EncodeI8(v)
	if len(f) != 2*len(v) || len(i) != len(v) {
		t.Fatal("size")
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
	for _, v := range [][]float32{{0.5, -0.5, 2.5, 127.6, -128.4, 300, -300}, {float32(math.Copysign(0, -1))}, {1e-3, 65519}} {
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

// scalarI8 is EncodeI8's definition.
func scalarI8(x float32) byte {
	if x != x {
		return simd.RoundI8(x) // NaN: whatever the scalar path does
	}
	return byte(int8(max(-128, min(127, math.Round(float64(x))))))
}

// roundI8 itself on integers, which the vector kernel takes instead on a
// host with AVX2: the scalar fallback must agree with the definition too.
func TestRoundI8ScalarOnIntegers(t *testing.T) {
	for i := -70000; i <= 70000; i++ {
		x := float32(i)
		if got, want := simd.RoundI8(x), scalarI8(x); got != want {
			t.Fatalf("simd.RoundI8(%v) = %d, want %d", x, int8(got), int8(want))
		}
	}
	for _, x := range []float32{2147483648, -2147483648, -2147483904, 1e30, -1e30, float32(math.Copysign(0, -1))} {
		if got, want := simd.RoundI8(x), scalarI8(x); got != want {
			t.Fatalf("simd.RoundI8(%v) = %d, want %d", x, int8(got), int8(want))
		}
	}
}
