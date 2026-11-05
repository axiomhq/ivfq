package rabitq

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/axiomhq/ivfq"
)

// TestGolden pins what stored columns and callers rely on: Seed, the
// rotation, the bytes MarshalBinary, MarshalRows, Select and AppendRows
// write from every representation of a column, and ScoreAll's scores and
// bounds. Every other test compares the package with itself; this one fails
// when the bytes change. If they must, re-encode stored columns and update
// want. Global scoring stays out: its ||q - c|| takes ivfq.Dot, whose float32
// sum order is the CPU's (GODEBUG=cpu.avx2=off moves its last bits).
func TestGolden(t *testing.T) {
	const want = "7b574535e4b2a5bed0dd2c64eda4346291307d6154001af27f3741f8b0bb046c"
	h := sha256.New()
	put := func(v any) { binary.Write(h, binary.LittleEndian, v) }
	putBytes := func(b []byte, err error) {
		if err != nil {
			t.Fatal(err)
		}
		h.Write(b)
	}
	put([]uint64{Seed(), Seed("ns", "vector"), Seed("a", "", "b")})
	for _, d := range []int{1, 3, 100, 129, 768} {
		v := corpus(1, d)[0]
		NewRotation(Seed("golden"), d).Apply(v)
		put(v)
	}
	for _, metric := range []ivfq.Metric{ivfq.L2, ivfq.Cosine} {
		for _, d := range []int{7, 129} {
			rot := NewRotation(Seed("golden"), d)
			v := corpus(70, d)
			v[5] = make([]float32, d)
			c := mustEncode(t, v, Options{Metric: metric, Rotation: rot})
			full, err := c.MarshalBinary()
			putBytes(full, err)
			rows, err := c.MarshalRows()
			putBytes(rows, err)
			packed := must(t)(UnmarshalBinaryBorrowed(full))
			packed.PackFastScanFrame(RotateCentroid(c.Code.Centroid, rot))
			for _, col := range []Quantizer{
				c,
				must(t)(UnmarshalBinary(full)),
				must(t)(UnmarshalBinaryBorrowed(full)),
				must(t)(UnmarshalRowsBorrowed(rows, c.Code.Centroid)),
				packed,
			} {
				putBytes(must(t)(col.Select([]int{69, 0, 5, 33})).MarshalBinary())
				// A zero vector and a row equal to the centroid, which stores no bits.
				extra := append(corpus(3, d), make([]float32, d), c.Code.Centroid)
				putBytes(must(t)(AppendRows(col, rot, extra)).MarshalBinary())
				q := NewQuery(corpus(2, d)[1], metric, rot)
				scores, bounds := make([]float32, col.Rows()), make([]float32, col.Rows())
				col.Scorer(q).ScoreAll(scores, bounds)
				put(scores)
				put(bounds)
			}
		}
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != want {
		t.Fatalf("golden hash %s, want %s", got, want)
	}
}
