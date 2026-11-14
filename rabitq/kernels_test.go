package rabitq

import (
	"bytes"
	"testing"

	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/internal/simd"
)

// kernelLevels are the vector kernels a column may be written or read
// with: the Go reference, AVX2, and AVX-512 where the CPU has it (on a CPU
// without it the third level is AVX2 again, and the test says so).
var kernelLevels = []struct {
	name         string
	avx2, avx512 bool
}{{"generic", false, false}, {"avx2", true, false}, {"avx512", true, true}}

// At every width, a column written with any kernel level and read with any
// other writes the same bytes and scores and bounds bit for bit as the Go
// reference writes and reads it. The codec's kernels (rotation, block sums,
// FastScan) are AVX2 or Go, each exact against the other, and it calls no
// AVX-512 kernel, so the avx512 level is the avx2 level here by design.
func TestCodesAcrossKernels(t *testing.T) {
	if avx2, avx512 := simd.Has(); !avx512 {
		t.Logf("this CPU has no AVX-512 (AVX2 %v): the avx512 level runs the AVX2 kernels", avx2)
	}
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, bits := range []int{0, 1, 3} {
			v := clusteredCorpus(300, 4, 128)
			v[7] = make([]float32, 128)
			queries := clusteredCorpus(5, 4, 128)
			cols := make([]Quantizer, len(kernelLevels))
			raw := make([][]byte, len(kernelLevels))
			for i, w := range kernelLevels {
				restore := simd.Limit(w.avx2, w.avx512)
				cols[i] = mustEncode(t, v, exOpts(metric, 128, bits))
				restore()
				var err error
				if raw[i], err = cols[i].MarshalBinary(); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(raw[i], raw[0]) {
					t.Errorf("%s %d bits: %s writes other bytes than generic", metric, bits+1, w.name)
				}
			}
			score := func(c Quantizer, q []float32, packed bool) (s, b []float32) {
				s, b = make([]float32, c.Rows()), make([]float32, c.Rows())
				if packed {
					data, err := c.MarshalBinary()
					if err != nil {
						t.Fatal(err)
					}
					c, _ = UnmarshalBinary(data)
					c.PackFastScan()
				}
				c.Scorer(NewQuery(q, metric, rotOf(c))).ScoreAll(s, b)
				return s, b
			}
			for _, q := range queries {
				for _, packed := range []bool{false, true} {
					restore := simd.Limit(false, false)
					ws, wb := score(cols[0], q, packed)
					restore()
					for wi, writer := range kernelLevels {
						for _, reader := range kernelLevels {
							restore := simd.Limit(reader.avx2, reader.avx512)
							gs, gb := score(cols[wi], q, packed)
							restore()
							for r := range gs {
								if gs[r] != ws[r] || gb[r] != wb[r] {
									t.Fatalf("%s %d bits packed=%v, written %s read %s, row %d: (%v, %v), generic (%v, %v)",
										metric, bits+1, packed, writer.name, reader.name, r, gs[r], gb[r], ws[r], wb[r])
								}
							}
						}
					}
				}
			}
		}
	}
}
