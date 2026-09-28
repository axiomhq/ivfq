package rabitq

import (
	"github.com/axiomhq/ivfq"
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
	opts := Options{Metric: ivfq.L2, Rotation: NewRotation(3, dims)}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Quantize(vectors, opts); err != nil {
			b.Fatal(err)
		}
	}
}
