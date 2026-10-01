package ivfq

// Centroid sets and deltas: a field's cluster centres and the slot upserts a
// split or recentre appends between full sets. Also the geometry of one
// cluster: its centroid and mean member distance.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
)

// ErrCorrupt wraps every decode failure of a centroid set or delta.
var ErrCorrupt = errors.New("ivfq: corrupt")

// EncodeCentroids writes a little-endian uint32 count, then each centroid's
// float32 values.
func EncodeCentroids(cs [][]float32) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(len(cs)))
	for _, c := range cs {
		binary.Write(&buf, binary.LittleEndian, c)
	}
	return buf.Bytes()
}

// DecodeCentroids reads what EncodeCentroids wrote, every centroid dims wide.
func DecodeCentroids(data []byte, dims int) ([][]float32, error) {
	// dims == 0 makes every entry zero-length, so a lying count prefix spins
	// 2^32 no-op reads instead of dying on a truncated one.
	if dims <= 0 {
		return nil, fmt.Errorf("%w: centroids need dims > 0, got %d", ErrCorrupt, dims)
	}
	r := bytes.NewReader(data)
	var k uint32
	if err := binary.Read(r, binary.LittleEndian, &k); err != nil {
		return nil, fmt.Errorf("%w: centroids: %v", ErrCorrupt, err)
	}
	// No capacity hint from k: the count prefix is untrusted bytes — a torn
	// write could claim 4 billion entries and OOM the pre-allocation.
	// Growing with actually-read data makes a lying prefix die on its first
	// truncated read instead.
	var out [][]float32
	for i := uint32(0); i < k; i++ {
		v := make([]float32, dims)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("%w: centroids: %v", ErrCorrupt, err)
		}
		out = append(out, v)
	}
	if r.Len() != 0 {
		return nil, fmt.Errorf("%w: trailing bytes in centroids", ErrCorrupt)
	}
	return out, nil
}

// CentroidUpsert is one entry of a centroid delta: the vector now at slot
// ID, which is either an existing slot (a recentred cluster) or one past the
// last (a cluster a split added), or, with a nil Vec, the set cut to its
// first ID slots (the last entry of a delta, after its upserts). A caller
// that drops a cluster moves the last one into its slot and cuts the end,
// so a merge is two entries and not a full set.
type CentroidUpsert struct {
	ID  int
	Vec []float32
}

const centroidDeltaMagic = "DCD\x01"

// deltaTruncate marks an entry's slot word as a truncation (no vector).
const deltaTruncate = 1 << 31

// EncodeCentroidDelta writes upserts as: magic, dims, count, then per entry
// the slot and the vector. The dims are carried so a delta applied to the
// wrong field fails on its header, not on a misaligned float.
func EncodeCentroidDelta(entries []CentroidUpsert) []byte {
	var buf bytes.Buffer
	buf.WriteString(centroidDeltaMagic)
	dims := 0
	for _, e := range entries {
		if e.Vec != nil {
			dims = len(e.Vec)
			break
		}
	}
	binary.Write(&buf, binary.LittleEndian, uint32(dims))
	binary.Write(&buf, binary.LittleEndian, uint32(len(entries)))
	for _, e := range entries {
		if e.Vec == nil {
			binary.Write(&buf, binary.LittleEndian, uint32(e.ID)|deltaTruncate)
			continue
		}
		binary.Write(&buf, binary.LittleEndian, uint32(e.ID))
		binary.Write(&buf, binary.LittleEndian, e.Vec)
	}
	return buf.Bytes()
}

// DecodeCentroidDelta reads what EncodeCentroidDelta wrote. Like
// DecodeCentroids it grows with bytes actually read, so a lying count dies
// on its first truncated entry instead of sizing an allocation.
func DecodeCentroidDelta(data []byte, dims int) ([]CentroidUpsert, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("%w: centroid delta needs dims > 0, got %d", ErrCorrupt, dims)
	}
	if len(data) < len(centroidDeltaMagic)+8 || string(data[:len(centroidDeltaMagic)]) != centroidDeltaMagic {
		return nil, fmt.Errorf("%w: centroid delta header", ErrCorrupt)
	}
	r := bytes.NewReader(data[len(centroidDeltaMagic):])
	var stored, count uint32
	if err := binary.Read(r, binary.LittleEndian, &stored); err != nil {
		return nil, fmt.Errorf("%w: centroid delta: %v", ErrCorrupt, err)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("%w: centroid delta: %v", ErrCorrupt, err)
	}
	if stored != 0 && int(stored) != dims {
		return nil, fmt.Errorf("%w: centroid delta is %d-wide, field is %d-wide", ErrCorrupt, stored, dims)
	}
	var out []CentroidUpsert
	for i := uint32(0); i < count; i++ {
		var id uint32
		if err := binary.Read(r, binary.LittleEndian, &id); err != nil {
			return nil, fmt.Errorf("%w: centroid delta: %v", ErrCorrupt, err)
		}
		if id&deltaTruncate != 0 {
			out = append(out, CentroidUpsert{ID: int(id &^ deltaTruncate)})
			continue
		}
		v := make([]float32, dims)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("%w: centroid delta: %v", ErrCorrupt, err)
		}
		out = append(out, CentroidUpsert{ID: int(id), Vec: v})
	}
	if r.Len() != 0 {
		return nil, fmt.Errorf("%w: trailing bytes in centroid delta", ErrCorrupt)
	}
	return out, nil
}

// ApplyCentroidDeltas folds deltas, oldest first, over a base set. The base
// slice is not mutated; its vectors are shared. A slot past the end of the
// set (a gap) is corruption: appends are contiguous by construction.
func ApplyCentroidDeltas(base [][]float32, deltas ...[]CentroidUpsert) ([][]float32, error) {
	out := slices.Clone(base)
	for _, d := range deltas {
		for _, e := range d {
			switch {
			case e.ID < 0 || e.ID > len(out):
				return nil, fmt.Errorf("%w: centroid delta slot %d past %d slots", ErrCorrupt, e.ID, len(out))
			case e.Vec == nil:
				out = out[:e.ID:e.ID]
			case e.ID == len(out):
				out = append(out, e.Vec)
			default:
				out[e.ID] = e.Vec
			}
		}
	}
	return out, nil
}

// CentroidDelta reports the upserts that turn prev into next when next
// keeps every slot of prev in place and only replaces or appends at most
// limit slots. Callers pass max(2, len(prev)/16): small enough that a delta
// stays cheap to apply on every load, and at least two so a split always
// qualifies. ok=false means write a full set: slots moved or vanished, or
// more than limit changed.
func CentroidDelta(prev, next [][]float32, limit int) ([]CentroidUpsert, bool) {
	if len(prev) == 0 || len(next) == 0 {
		return nil, false
	}
	var out []CentroidUpsert
	for i, v := range next {
		if i < len(prev) && slices.Equal(prev[i], v) {
			continue
		}
		if len(out) == limit {
			return nil, false
		}
		out = append(out, CentroidUpsert{ID: i, Vec: v})
	}
	if len(next) < len(prev) {
		out = append(out, CentroidUpsert{ID: len(next)})
	}
	return out, true
}

// ApplyCentroidDeltas upserts deltas, oldest first, into t in place.
func (t *Tree) ApplyCentroidDeltas(deltas [][]CentroidUpsert) error {
	for _, d := range deltas {
		for _, e := range d {
			var err error
			if e.Vec == nil {
				err = t.Truncate(e.ID)
			} else {
				err = t.Upsert(e.ID, e.Vec)
			}
			if err != nil {
				return fmt.Errorf("%w: centroid delta: %v", ErrCorrupt, err)
			}
		}
	}
	return nil
}

// UpsertAll turns t into centroids' tree without a Build: t itself when
// they already agree, and otherwise a clone with the changed and appended
// slots upserted — what ApplyCentroidDeltas does with a delta. A split
// changes one slot and appends one, and rebuilding a ~10k-centroid tree for
// that costs far more than two upserts. A shorter set cuts the tree to
// its length first (Truncate). At most limit slots are upserted (callers
// pass max(2, t.Leaves()/16), the CentroidDelta bound); past it ok is
// false and the caller builds.
func (t *Tree) UpsertAll(centroids [][]float32, limit int) (*Tree, bool) {
	leaves := t.Leaves()
	if len(centroids) == 0 {
		return nil, false
	}
	var changed []int
	for i := range centroids {
		if i < leaves && slices.Equal(t.Leaf(i), centroids[i]) {
			continue
		}
		if len(changed) == limit {
			return nil, false
		}
		changed = append(changed, i)
	}
	if len(changed) == 0 && len(centroids) == leaves {
		return t, true
	}
	c := t.Clone()
	if len(centroids) < leaves {
		if err := c.Truncate(len(centroids)); err != nil {
			return nil, false
		}
	}
	for _, i := range changed {
		if err := c.Upsert(i, centroids[i]); err != nil {
			return nil, false
		}
	}
	return &c, true
}

// Centroid is the mean of the dims-wide vectors (spherical: normalized) and
// their mean distance to it; nil for none. Vectors of another width are
// skipped.
func Centroid(vecs [][]float32, dims int, spherical bool) ([]float32, float64) {
	sum := make([]float64, dims)
	n := 0
	for _, v := range vecs {
		if len(v) != dims {
			continue
		}
		for i, x := range v {
			sum[i] += float64(x)
		}
		n++
	}
	if n == 0 {
		return nil, 0
	}
	c := make([]float32, dims)
	for i := range c {
		c[i] = float32(sum[i] / float64(n))
	}
	if spherical {
		Normalize(c)
	}
	return c, MeanDistance(vecs, c)
}

// MeanDistance is the mean Euclidean distance from c to the vectors as wide
// as c; 0 for none.
func MeanDistance(vecs [][]float32, c []float32) float64 {
	var r float64
	n := 0
	for _, v := range vecs {
		if len(v) != len(c) {
			continue
		}
		r += math.Sqrt(float64(L2Sq(c, v)))
		n++
	}
	if n == 0 {
		return 0
	}
	return r / float64(n)
}
