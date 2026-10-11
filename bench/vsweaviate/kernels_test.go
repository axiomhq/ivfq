package vsweaviate

// Kernel benchmarks: ivfq/internal/simd + go-simd/floats against weaviate's
// hnsw/distancer/asm. Vectors come from a pool larger than L1 but inside L2,
// so every op reads fresh lines the way a scan does, not one hot pair.

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"testing"

	"github.com/axiomhq/ivfq/internal/simd"
	"github.com/go-simd/floats"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer/asm"
	"golang.org/x/sys/cpu"
)

const pool = 256

var dimsList = []int{128, 384, 768, 1536}

var sinkF float32
var sinkU uint64

func randF32(n, d int, r *rand.Rand) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, d)
		for j := range out[i] {
			out[i][j] = float32(r.NormFloat64())
		}
	}
	return out
}

func randU8(n, d int, r *rand.Rand) [][]uint8 {
	out := make([][]uint8, n)
	for i := range out {
		out[i] = make([]uint8, d)
		for j := range out[i] {
			out[i][j] = uint8(r.IntN(256))
		}
	}
	return out
}

// pairs runs f over pool pairs, b.N times, SetBytes = bytes read per call.
func pairs(b *testing.B, bytesPerOp int, f func(i int)) {
	b.SetBytes(int64(bytesPerOp))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f(i & (pool - 1))
	}
}

func near(a, b, rel float64) bool {
	return math.Abs(a-b) <= rel*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

// TestKernelsAgree pins every pair of kernels we compare to the same value,
// so a speed difference is never a different computation.
func TestKernelsAgree(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randF32(2, d, r)[0], randF32(2, d, r)[1]
		if a, b := float64(simd.Dot(x, y)), float64(asm.DotAVX512(x, y)); !near(a, b, 1e-4) {
			t.Errorf("dot d=%d: ivfq %v weaviate %v", d, a, b)
		}
		if a, b := float64(simd.L2Sq(x, y)), float64(asm.L2AVX512(x, y)); !near(a, b, 1e-4) {
			t.Errorf("l2 d=%d: ivfq %v weaviate %v", d, a, b)
		}
		u, v := randU8(2, d, r)[0], randU8(2, d, r)[1]
		var want uint32
		for i := range u {
			want += uint32(u[i]) * uint32(v[i])
		}
		if got := asm.DotByteVNNI512(u, v); got != want {
			t.Errorf("u8 vnni d=%d: %d want %d", d, got, want)
		}
		w := d / 64
		a, c := make([]uint64, w), make([]uint64, w)
		for i := range a {
			a[i], c[i] = r.Uint64(), r.Uint64()
		}
		var h int
		for i := range a {
			h += bits.OnesCount64(a[i] ^ c[i])
		}
		if got := asm.HammingBitwiseAVX512(a, c); int(got) != h {
			t.Errorf("hamming d=%d: %v want %d", d, got, h)
		}
		planes := make([]uint64, 4*w)
		for i := range planes {
			planes[i] = r.Uint64()
		}
		ones, wt := simd.BitProductWords(a, planes)
		var wo, ww int
		for i := range a {
			wo += bits.OnesCount64(a[i])
			for p := 0; p < 4; p++ {
				ww += bits.OnesCount64(a[i]&planes[p*w+i]) << p
			}
		}
		if int(ones) != wo || int(wt) != ww {
			t.Errorf("bitproduct d=%d: %d,%d want %d,%d", d, ones, wt, wo, ww)
		}
	}
}

func BenchmarkDotF32(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randF32(pool, d, r), randF32(pool, d, r)
		prov := distancer.NewDotProductProvider()
		run := func(name string, f func(a, b []float32) float32) {
			b.Run(fmt.Sprintf("d=%d/%s", d, name), func(b *testing.B) {
				var s float32
				pairs(b, 8*d, func(i int) { s += f(x[i], y[i]) })
				sinkF = s
			})
		}
		run("ivfq_floats", simd.Dot)
		run("ivfq_floats_raw", floats.Float32Dot)
		run("wv_avx256", asm.DotAVX256)
		if cpu.X86.HasAVX512F {
			run("wv_avx512", asm.DotAVX512)
		}
		run("wv_provider", func(a, c []float32) float32 { v, _ := prov.SingleDist(a, c); return v })
	}
}

func BenchmarkL2F32(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randF32(pool, d, r), randF32(pool, d, r)
		run := func(name string, f func(a, b []float32) float32) {
			b.Run(fmt.Sprintf("d=%d/%s", d, name), func(b *testing.B) {
				var s float32
				pairs(b, 8*d, func(i int) { s += f(x[i], y[i]) })
				sinkF = s
			})
		}
		run("ivfq_L2Sq", simd.L2Sq)
		run("wv_avx256", asm.L2AVX256)
		if cpu.X86.HasAVX512F {
			run("wv_avx512", asm.L2AVX512)
		}
	}
}

// u8 x u8. ivfq has no u8 kernel; its integer kernel is int8 (DotsI8, a
// row-major block against one query, AVX2). Weaviate's rows are one pair per
// call. ns/vec is per scored vector for both.
func BenchmarkDotU8(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randU8(pool, d, r), randU8(pool, d, r)
		run := func(name string, f func(a, b []uint8) uint32) {
			b.Run(fmt.Sprintf("d=%d/%s", d, name), func(b *testing.B) {
				var s uint32
				pairs(b, 2*d, func(i int) { s += f(x[i], y[i]) })
				sinkU += uint64(s)
			})
		}
		if cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512BW {
			run("wv_vnni512", asm.DotByteVNNI512)
		}
		run("wv_avx256", asm.DotByteAVX256)

		const rows = 64
		q := make([]int8, d)
		blk := make([]int8, rows*d)
		for i := range q {
			q[i] = int8(r.IntN(256))
		}
		for i := range blk {
			blk[i] = int8(r.IntN(256))
		}
		out := make([]int32, rows)
		b.Run(fmt.Sprintf("d=%d/ivfq_DotsI8(int8,block64)", d), func(b *testing.B) {
			b.SetBytes(int64(rows * d))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				simd.DotsI8(q, blk, out)
			}
			sinkU += uint64(out[0])
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*rows), "ns/vec")
		})
	}
}

func BenchmarkL2U8(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randU8(pool, d, r), randU8(pool, d, r)
		run := func(name string, f func(a, b []uint8) uint32) {
			b.Run(fmt.Sprintf("d=%d/%s", d, name), func(b *testing.B) {
				var s uint32
				pairs(b, 2*d, func(i int) { s += f(x[i], y[i]) })
				sinkU += uint64(s)
			})
		}
		if cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512BW {
			run("wv_vnni512", asm.L2ByteVNNI512)
		}
		run("wv_avx256", asm.L2ByteAVX256)
	}
}

// float32 query x byte row. ivfq: DotF32I8 (fused) and the older DecodeI8+Dot.
// Weaviate reads the bytes as uint8, ivfq as int8; the work is the same.
func BenchmarkDotFloatByte(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		x, y := randF32(pool, d, r), randU8(pool, d, r)
		b.Run(fmt.Sprintf("d=%d/wv_fused_avx256", d), func(b *testing.B) {
			var s float32
			pairs(b, 5*d, func(i int) { s += asm.DotFloatByteAVX256(x[i], y[i]) })
			sinkF = s
		})
		b.Run(fmt.Sprintf("d=%d/ivfq_DotF32I8_fused", d), func(b *testing.B) {
			var s float32
			pairs(b, 5*d, func(i int) { s += simd.DotF32I8(x[i], y[i]) })
			sinkF = s
		})
		tmp := make([]float32, d)
		b.Run(fmt.Sprintf("d=%d/ivfq_DecodeI8+Dot", d), func(b *testing.B) {
			var s float32
			pairs(b, 5*d, func(i int) {
				simd.DecodeI8(tmp, y[i])
				s += simd.Dot(x[i], tmp)
			})
			sinkF = s
		})
	}
}

// 1-bit code against the query's bit planes: the inner loop of a 1-bit scan.
// ivfq: BitProductWords over four planes (ones + weighted, one pass).
// weaviate: five Hamming distances, one per plane (HammingBitwiseAVX*), the
// way BinaryRQDistancer.Distance does it at >= 512 bits; below that it uses
// math/bits (wv_popcnt).
func BenchmarkBitScan(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range dimsList {
		w := d / 64
		rows := make([][]uint64, pool)
		for i := range rows {
			rows[i] = make([]uint64, w)
			for j := range rows[i] {
				rows[i][j] = r.Uint64()
			}
		}
		planes := make([]uint64, 5*w)
		for i := range planes {
			planes[i] = r.Uint64()
		}
		b.Run(fmt.Sprintf("d=%d/ivfq_BitProduct(4planes)", d), func(b *testing.B) {
			var s uint64
			pairs(b, d/8, func(i int) {
				o, wt := simd.BitProductWords(rows[i], planes[:4*w])
				s += o + wt
			})
			sinkU += s
		})
		hamm := func(name string, f func(a, c []uint64) float32) {
			b.Run(fmt.Sprintf("d=%d/%s(5planes)", d, name), func(b *testing.B) {
				var s float32
				pairs(b, d/8, func(i int) {
					for p := 0; p < 5; p++ {
						s += f(planes[p*w:(p+1)*w], rows[i])
					}
				})
				sinkF = s
			})
		}
		hamm("wv_hamming_avx256", asm.HammingBitwiseAVX256)
		if cpu.X86.HasAVX512F {
			hamm("wv_hamming_avx512", asm.HammingBitwiseAVX512)
		}
		hamm("wv_hamming_popcnt", func(a, c []uint64) float32 {
			var n int
			for i := range a {
				n += bits.OnesCount64(a[i] ^ c[i])
			}
			return float32(n)
		})
	}
}

// TestKernelsAgreeAllLengths runs our kernels and weaviate's side by side on
// every length 1..140 and a few large ones, with large magnitudes and the
// extreme byte values: float kernels within float32 reassociation error,
// integer kernels exactly. Bytes for DotF32I8 are kept below 128, where
// weaviate's uint8 and our int8 reading are the same number.
func TestKernelsAgreeAllLengths(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	lens := []int{255, 256, 257, 768, 1000, 1536, 4097}
	for n := 1; n <= 140; n++ {
		lens = append(lens, n)
	}
	for _, n := range lens {
		x, y := make([]float32, n), make([]float32, n)
		var dot, l2, magD, magL float64
		for i := range x {
			x[i], y[i] = float32(r.NormFloat64()*1000), float32(r.NormFloat64()*1000)
			p, d := float64(x[i])*float64(y[i]), float64(x[i])-float64(y[i])
			dot, l2, magD, magL = dot+p, l2+d*d, magD+math.Abs(p), magL+d*d
		}
		for name, got := range map[string][2]float64{
			"dot": {float64(simd.Dot(x, y)), float64(asm.DotAVX512(x, y))},
			"l2":  {float64(simd.L2Sq(x, y)), float64(asm.L2AVX512(x, y))},
		} {
			want, mag := dot, magD
			if name == "l2" {
				want, mag = l2, magL
			}
			for k, g := range got {
				if math.Abs(g-want) > 2e-5*mag {
					t.Errorf("%s n=%d kernel %d: %v, want %v", name, n, k, g, want)
				}
			}
		}

		b := make([]byte, n)
		for i := range b {
			b[i] = byte(r.IntN(128))
		}
		var fb float64
		for i := range x {
			fb += float64(x[i]) * float64(b[i])
		}
		if a, w := float64(simd.DotF32I8(x, b)), float64(asm.DotFloatByteAVX256(x, b)); math.Abs(a-fb) > 2e-5*magD*0.2+1e-3*float64(n) || math.Abs(a-w) > 1e-3*float64(n)+2e-5*math.Abs(fb) {
			t.Errorf("f32xi8 n=%d: ours %v weaviate %v exact %v", n, a, w, fb)
		}

		// int8 block dot against the exact sum, every value extreme.
		q := make([]int8, n)
		blk := make([]int8, 3*n)
		for i := range q {
			q[i] = []int8{-128, 127, int8(r.IntN(256) - 128)}[r.IntN(3)]
		}
		for i := range blk {
			blk[i] = []int8{-128, 127, int8(r.IntN(256) - 128)}[r.IntN(3)]
		}
		out := make([]int32, 3)
		simd.DotsI8(q, blk, out)
		for row := range out {
			var want int32
			for j := range q {
				want += int32(q[j]) * int32(blk[row*n+j])
			}
			if out[row] != want {
				t.Errorf("DotsI8 n=%d row %d: %d, want %d", n, row, out[row], want)
			}
		}
	}
}
