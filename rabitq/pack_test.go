package rabitq

import (
	"bytes"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"slices"
	"testing"

	"github.com/axiomhq/ivfq"
)

// packRowAtATime is PackFastScanFrame as it was through v0.15.0, a row at a
// time: the reference pack.go must reproduce bit for bit. rc nil is
// PackFastScan.
func packRowAtATime(c *Quantizer, rc []float32) {
	b := c.Code
	if b.fast != nil && (rc == nil || b.fast.xbc != nil) {
		return
	}
	d := b.width()
	var xbc []float32
	var rcc float64
	if rc != nil && len(rc) == d && fastGroups(d) != 0 {
		rows, words := b.Rows(), bitWords(d)
		var sum float64
		for _, v := range rc {
			sum += float64(v)
		}
		scale := 1 / math.Sqrt(float64(d))
		xbc = make([]float32, rows)
		for r := range rows {
			var on float64
			for w := range words {
				x := b.word(r*words + w)
				for x != 0 {
					j := w*64 + bits.TrailingZeros64(x)
					on += float64(rc[j])
					x &= x - 1
				}
			}
			xbc[r] = float32((2*on - sum) * scale)
		}
		for _, v := range rc {
			rcc += float64(v) * float64(v)
		}
	}
	defer func() {
		if xbc != nil && b.fast != nil {
			b.fast.xbc, b.fast.rc, b.fast.rcc = xbc, rc, rcc
		}
	}()
	groups := fastGroups(b.width())
	if b.fast != nil || groups == 0 {
		return
	}
	rows, words := b.Rows(), bitWords(b.width())
	f := &fastLayout{groups: groups, blocks: (rows + 31) / 32, ones: make([]uint16, rows),
		norms: make([]float32, rows), aligns: make([]float32, rows)}
	f.nibs = make([]byte, f.blocks*groups*16)
	for r := range rows {
		k, i, shift := r/32, r%32, uint(0)
		if i >= 16 {
			i, shift = i-16, 4
		}
		ones := 0
		for w := range words {
			x := b.word(r*words + w)
			ones += bits.OnesCount64(x)
			for n := range 16 {
				g := w*16 + n
				if g >= groups {
					break
				}
				f.nibs[(k*groups+g)*16+i] |= byte(x>>(4*n)&0xF) << shift
			}
		}
		f.ones[r] = uint16(ones)
		f.norms[r], f.aligns[r] = b.norm(r), b.align(r)
	}
	if b.borrowed != nil && b.cOff > 0 {
		c := make([]float32, b.dims)
		for i := range c {
			c[i] = b.centroid(i)
		}
		b.Centroid, f.ownCentroid = c, true
	}
	b.Norms, b.Aligns = f.norms, f.aligns
	b.Words, b.borrowed, b.cOff = nil, nil, 0
	b.fast = f
}

// samePacked fails unless got and want are packed identically: layout,
// popcounts, scalars and frame sums bit for bit, and the centroid.
func samePacked(t *testing.T, name string, got, want *Code) {
	t.Helper()
	g, w := got.fast, want.fast
	if (g == nil) != (w == nil) {
		t.Fatalf("%s: packed %v, reference packed %v", name, g != nil, w != nil)
	}
	if g == nil {
		return
	}
	if g.groups != w.groups || g.blocks != w.blocks || !bytes.Equal(g.nibs, w.nibs) || !slices.Equal(g.ones, w.ones) {
		t.Fatalf("%s: integer layout differs", name)
	}
	for _, p := range [][2][]float32{{g.norms, w.norms}, {g.aligns, w.aligns}, {g.xbc, w.xbc}, {got.Norms, want.Norms}, {got.Aligns, want.Aligns}, {got.Centroid, want.Centroid}} {
		if len(p[0]) != len(p[1]) || (p[0] == nil) != (p[1] == nil) {
			t.Fatalf("%s: lengths %d, %d", name, len(p[0]), len(p[1]))
		}
		for i := range p[0] {
			if math.Float32bits(p[0][i]) != math.Float32bits(p[1][i]) {
				t.Fatalf("%s: float %d is %v (%#x), reference %v (%#x)", name, i, p[0][i], math.Float32bits(p[0][i]), p[1][i], math.Float32bits(p[1][i]))
			}
		}
	}
	if math.Float64bits(g.rcc) != math.Float64bits(w.rcc) || len(g.rc) != len(w.rc) || g.ownCentroid != w.ownCentroid ||
		got.Words != nil || got.borrowed != nil || got.cOff != 0 || got.retained() != want.retained() {
		t.Fatalf("%s: column state differs", name)
	}
}

// randomCode is a column of rows random bit rows dims wide (padding clear),
// a zero row at 5, around centroid, with scalars the decoders accept.
func randomCode(rng *rand.Rand, rows, dims int, centroid []float32, density float64) Quantizer {
	w := bitWords(dims)
	b := &Code{Words: make([]uint64, rows*w), Centroid: centroid, Norms: make([]float32, rows), Aligns: make([]float32, rows), Seed: 7}
	for r := range rows {
		if r == 5 {
			continue
		}
		for j := range dims {
			if rng.Float64() < density {
				b.Words[r*w+j/64] |= 1 << (j % 64)
			}
		}
		floor := 1 / math.Sqrt(float64(dims)) // the decoder's floor is 0.999 of it
		b.Norms[r], b.Aligns[r] = float32(1+rng.Float64()), float32(floor+(1-floor)*rng.Float64())
	}
	return Quantizer{Dims: dims, Code: b}
}

// adversarialRC is a rotated centroid whose float64 sums depend on their
// order: huge values of either sign beside tiny and subnormal ones, -0,
// and ordinary ones.
func adversarialRC(rng *rand.Rand, dims int) []float32 {
	rc := make([]float32, dims)
	for j := range rc {
		switch rng.Intn(7) {
		case 0:
			rc[j] = float32(rng.NormFloat64()) * 1e30
		case 1:
			rc[j] = math.SmallestNonzeroFloat32 * float32(rng.Intn(9)-4)
		case 2:
			rc[j] = float32(math.Copysign(0, -1))
		case 3:
			rc[j] = float32(rng.NormFloat64()) * 1e-20
		case 4:
			rc[j] = math.MaxFloat32 / 4 * float32(rng.Intn(3)-1)
		default:
			rc[j] = float32(rng.NormFloat64() * 100)
		}
	}
	return rc
}

// The block packer is the row-at-a-time one, bit for bit: widths odd, not
// a multiple of 4, 64 or 128, up to 1,024 and past it; row counts either
// side of a 16- and 32-row boundary (partial blocks); typed, decoded,
// borrowed with its own centroid and borrowed against a shared one; zero,
// signed and adversarial centroids and rotated centroids; dense, sparse
// and empty bits; PackFastScan, PackFastScanFrame, and a frame after a
// plain pack.
func TestPackMatchesRowAtATime(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, dims := range []int{1, 3, 4, 7, 63, 64, 65, 100, 127, 128, 130, 191, 768, 1000, 1024, 1536, 4096, 4097} {
		for _, rows := range []int{0, 1, 5, 6, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100} {
			for ci, cent := range []string{"zero", "signed", "adversarial"} {
				centroid := make([]float32, dims)
				var rc []float32
				switch cent {
				case "zero":
					rc = make([]float32, dims)
				case "signed":
					for j := range centroid {
						centroid[j] = float32(rng.NormFloat64())
					}
					rc = RotateCentroid(centroid, NewRotation(7, dims))
				default:
					rc = adversarialRC(rng, dims)
					for j := range centroid {
						centroid[j] = float32(rng.NormFloat64() * 1e6)
					}
				}
				density := []float64{0.5, 0.03, 0.97}[ci]
				src := randomCode(rng, rows, dims, centroid, density)
				full, err := src.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				part, err := src.MarshalRows()
				if err != nil {
					t.Fatal(err)
				}
				decoders := map[string]func() Quantizer{
					"typed": func() Quantizer {
						b := *src.Code
						b.Words, b.Norms, b.Aligns = slices.Clone(b.Words), slices.Clone(b.Norms), slices.Clone(b.Aligns)
						return Quantizer{Dims: dims, Code: &b}
					},
					"decoded":  func() Quantizer { return must(t)(UnmarshalBinary(full)) },
					"borrowed": func() Quantizer { return must(t)(UnmarshalBinaryBorrowed(bytes.Clone(full))) },
					"rows":     func() Quantizer { return must(t)(UnmarshalRowsBorrowed(bytes.Clone(part), centroid)) },
				}
				for kind, dec := range decoders {
					for _, mode := range []string{"plain", "frame", "plain+frame"} {
						name := fmt.Sprintf("d%d/r%d/%s/%s/%s", dims, rows, cent, kind, mode)
						got, want := dec(), dec()
						if got.Code == nil || want.Code == nil {
							t.Fatalf("%s: did not decode", name)
						}
						switch mode {
						case "plain":
							got.PackFastScan()
							packRowAtATime(&want, nil)
						case "frame":
							got.PackFastScanFrame(rc)
							packRowAtATime(&want, rc)
						default:
							got.PackFastScan()
							got.PackFastScanFrame(rc)
							packRowAtATime(&want, nil)
							packRowAtATime(&want, rc)
						}
						samePacked(t, name, got.Code, want.Code)
						if mb, err := got.MarshalBinary(); err != nil || !bytes.Equal(mb, full) {
							t.Fatalf("%s: packed column marshals differently (%v)", name, err)
						}
					}
				}
			}
		}
	}
}

func must(t *testing.T) func(Quantizer, error) Quantizer {
	return func(q Quantizer, err error) Quantizer {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return q
	}
}

// On real quantized columns the two packers agree too, and a global
// query scores and bounds through either bit for bit.
func TestPackScoresAsRowAtATime(t *testing.T) {
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, dims := range []int{7, 128, 130, 768, 1024} {
			for _, rows := range []int{1, 33, 64, 100} {
				v := offCentre(rows, dims, int64(dims+rows))
				if rows > 5 {
					v[5] = make([]float32, dims)
				}
				rot := NewRotation(17, dims)
				c, err := Quantize(v, Options{Metric: metric, Rotation: rot})
				if err != nil {
					t.Fatal(err)
				}
				part, err := c.MarshalRows()
				if err != nil {
					t.Fatal(err)
				}
				rc := RotateCentroid(c.Code.Centroid, rot)
				got, err := UnmarshalRowsBorrowed(part, c.Code.Centroid)
				if err != nil {
					t.Fatal(err)
				}
				want, _ := UnmarshalRowsBorrowed(part, c.Code.Centroid)
				got.PackFastScanFrame(rc)
				packRowAtATime(&want, rc)
				name := fmt.Sprintf("%v/d%d/r%d", metric, dims, rows)
				samePacked(t, name, got.Code, want.Code)
				for _, exact := range []bool{false, true} {
					pq := NewQuery(offCentre(1, dims, 99)[0], metric, rot).Global(rot)
					pq.Exact = exact
					var sg, sw Scorer
					if !sg.BindGlobal(&got, pq) || !sw.BindGlobal(&want, pq) {
						t.Fatalf("%s: did not bind", name)
					}
					gs, gb, ws, wb := make([]float32, rows), make([]float32, rows), make([]float32, rows), make([]float32, rows)
					sg.ScoreAll(gs, gb)
					sw.ScoreAll(ws, wb)
					for r := range rows {
						if math.Float32bits(gs[r]) != math.Float32bits(ws[r]) || math.Float32bits(gb[r]) != math.Float32bits(wb[r]) {
							t.Fatalf("%s row %d: %v/%v, reference %v/%v", name, r, gs[r], gb[r], ws[r], wb[r])
						}
					}
				}
			}
		}
	}
}

func TestTranspose8(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for range 1000 {
		var m, want [8]uint64
		for i := range m {
			m[i] = rng.Uint64()
		}
		for i := range 8 {
			for j := range 8 {
				want[j] |= (m[i] >> (8 * j) & 0xFF) << (8 * i)
			}
		}
		transpose8(&m)
		if m != want {
			t.Fatalf("got %x, want %x", m, want)
		}
	}
}

func TestPackFixtureMatchesRowAtATime(t *testing.T) {
	centroid, rc, blocks := packFixture(t)
	if blocks == nil {
		t.Skip("IVFQ_PACK_FIXTURE not set")
	}
	for i, raw := range blocks {
		got, err := UnmarshalRowsBorrowed(raw, centroid)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := UnmarshalRowsBorrowed(raw, centroid)
		got.PackFastScanFrame(rc)
		packRowAtATime(&want, rc)
		samePacked(t, fmt.Sprintf("fixture block %d (%dx%d)", i, got.Rows(), got.Dims), got.Code, want.Code)
	}
}

// BenchmarkPackReference is BenchmarkDecodePack's frame case through the
// row-at-a-time reference, in the same binary.
func BenchmarkPackReference(b *testing.B) {
	for _, dims := range []int{128, 768, 1024} {
		centroid, rc, part := packBenchColumn(b, dims, 64)
		b.Run(fmt.Sprintf("frame/d%d/r64", dims), func(b *testing.B) {
			for b.Loop() {
				q, err := UnmarshalRowsBorrowed(part, centroid)
				if err != nil {
					b.Fatal(err)
				}
				packRowAtATime(&q, rc)
			}
		})
	}
}
