package ivfq

import (
	"fmt"
	"testing"
)

func BenchmarkQuantizeL2(b *testing.B) {
	const rows, dims = 1024, 128
	vectors := make([][]float32, rows)
	for i := range vectors {
		v := make([]float32, dims)
		for j := range v {
			v[j] = float32((i*31+j*17)%503-251) / 8
		}
		vectors[i] = v
	}
	opts := Options{Dims: dims, Metric: "l2", Seed: 3}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Quantize(vectors, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBitProduct is one row's five popcounts: the dispatched kernel
// against the generic loop, at the widths that matter.
func BenchmarkBitProduct(b *testing.B) {
	for _, dims := range []int{128, 768, 1536} {
		words := bitWords(dims)
		row := make([]uint64, words)
		planes := make([]uint64, 4*words)
		for i := range row {
			row[i] = uint64(i)*0x9E3779B97F4A7C15 ^ 0x5555
		}
		for i := range planes {
			planes[i] = uint64(i)*0xC2B2AE3D27D4EB4F ^ 0x3333
		}
		b.Run(fmt.Sprintf("generic/%d", dims), func(b *testing.B) {
			var s uint64
			for b.Loop() {
				o, w := bitProductGeneric(row, planes)
				s += o + w
			}
			_ = s
		})
		b.Run(fmt.Sprintf("kernel/%d", dims), func(b *testing.B) {
			var s uint64
			for b.Loop() {
				o, w := bitProductWords(row, planes)
				s += o + w
			}
			_ = s
		})
	}
}
