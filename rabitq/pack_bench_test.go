package rabitq

import (
	"fmt"
	"testing"

	"github.com/axiomhq/ivfq"
)

// packBenchColumn is a rows x dims l2 column around an off-centre corpus's
// mean, as MarshalRows writes it, with its centroid and rotated centroid.
func packBenchColumn(tb testing.TB, dims, rows int) (centroid, rc []float32, part []byte) {
	tb.Helper()
	rot := NewRotation(Seed("bench", "pack"), dims)
	c, err := Quantize(offCentre(rows, dims, int64(dims)), Options{Metric: ivfq.L2, Rotation: rot})
	if err != nil {
		tb.Fatal(err)
	}
	if part, err = c.MarshalRows(); err != nil {
		tb.Fatal(err)
	}
	return c.Code.Centroid, RotateCentroid(c.Code.Centroid, rot), part
}

// BenchmarkDecodePack is what a cold hood read costs per 64-row block once
// its bytes are read: UnmarshalRowsBorrowed, then PackFastScan or, as a
// reader that serves global queries does, PackFastScanFrame.
func BenchmarkDecodePack(b *testing.B) {
	for _, dims := range []int{128, 768, 1024} {
		centroid, rc, part := packBenchColumn(b, dims, 64)
		for _, frame := range []bool{false, true} {
			name := fmt.Sprintf("plain/d%d/r64", dims)
			if frame {
				name = fmt.Sprintf("frame/d%d/r64", dims)
			}
			b.Run(name, func(b *testing.B) {
				b.SetBytes(int64(len(part)))
				for b.Loop() {
					q, err := UnmarshalRowsBorrowed(part, centroid)
					if err != nil {
						b.Fatal(err)
					}
					if frame {
						q.PackFastScanFrame(rc)
					} else {
						q.PackFastScan()
					}
				}
			})
		}
	}
}
