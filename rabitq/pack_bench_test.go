package rabitq

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/axiomhq/ivfq"
)

// The decode-and-pack benchmarks use only the package's public API and
// offCentre, so this file runs unchanged against a release without
// pack.go: a baseline and a treatment are the same benchmark.

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

// packFixture is IVFQ_PACK_FIXTURE, a directory of a real hood: its
// centroid as little-endian float32s (centroid.f32), its rotated centroid
// (rc.f32, optional: RotateCentroid otherwise) and one or more code
// blocks as MarshalRows wrote them (*.dqr). Nil when unset.
func packFixture(tb testing.TB) (centroid, rc []float32, blocks [][]byte) {
	dir := os.Getenv("IVFQ_PACK_FIXTURE")
	if dir == "" {
		return nil, nil, nil
	}
	f32s := func(name string) []float32 {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out
	}
	centroid, rc = f32s("centroid.f32"), f32s("rc.f32")
	names, err := filepath.Glob(filepath.Join(dir, "*.dqr"))
	if err != nil || len(names) == 0 || centroid == nil {
		tb.Fatalf("IVFQ_PACK_FIXTURE %s: want centroid.f32 and *.dqr (%v)", dir, err)
	}
	slices.Sort(names)
	for _, n := range names {
		raw, err := os.ReadFile(n)
		if err != nil {
			tb.Fatal(err)
		}
		blocks = append(blocks, raw)
	}
	if rc == nil {
		q, err := UnmarshalRowsBorrowed(blocks[0], centroid)
		if err != nil {
			tb.Fatal(err)
		}
		rc = RotateCentroid(centroid, NewRotation(q.Code.Seed, q.Dims))
	}
	return centroid, rc, blocks
}

// BenchmarkDecodePackFixture is BenchmarkDecodePack's frame case over a
// real hood's blocks (IVFQ_PACK_FIXTURE), one op per block decoded and
// packed in turn.
func BenchmarkDecodePackFixture(b *testing.B) {
	centroid, rc, blocks := packFixture(b)
	if blocks == nil {
		b.Skip("IVFQ_PACK_FIXTURE not set")
	}
	var bytes int
	for _, raw := range blocks {
		bytes += len(raw)
	}
	b.SetBytes(int64(bytes / len(blocks)))
	i := 0
	for b.Loop() {
		q, err := UnmarshalRowsBorrowed(blocks[i], centroid)
		if err != nil {
			b.Fatal(err)
		}
		q.PackFastScanFrame(rc)
		if i++; i == len(blocks) {
			i = 0
		}
	}
}
