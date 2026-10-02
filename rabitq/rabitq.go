package rabitq

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/axiomhq/ivfq"
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/bits"
)

// The 1-bit candidate codec: RaBitQ (Gao & Long, SIGMOD 2024,
// arXiv:2405.12497), the "1-bit codes, 16-32x compression" of
// the current generation of ANN indexes.
//
// One row of a hood is:
//
//	x   = P^T (o - c) / ||o - c||     the rotated, normalized residual
//	bit = [x_j >= 0] for each j       ONE bit per dimension
//	x_b = (2*bit - 1) / sqrt(D)       the unit vector those bits stand for
//
// plus two float32 scalars: ||o - c||, and <x_b, x> (how well the sign
// pattern points at the residual; call it the row's ALIGNMENT). At 128 dims
// that is 16 + 8 = 24 bytes against the 512 bytes of the f32 vector, 21x,
// and against int8's 128 + 16 = 144 bytes, 6x. A probed cluster is one
// read of its column, and this is how big that read is.
//
// The query side rotates and normalizes ITS residual to the same centroid,
// quantizes it to 4 bits per dimension (the paper's B_q), and estimates
//
//	<x, u> ~= <x_b, u~> / <x_b, x>
//
// with four bit-plane popcounts per row. The paper proves that estimator
// unbiased and gives it an error bound of
//
//	sqrt(1 - <x_b,x>^2) / <x_b,x> * eps0 / sqrt(D-1)
//
// at confidence 1 - 2exp(-c0*eps0^2). This implementation adds the query's
// own rounding residual to that sigma (it is measured exactly, per query,
// rather than bounded worst-case; see bitScorer) and reports eps0 sigmas as
// the bound. THE BOUND IS PROBABILISTIC: unlike the int8 codec's triangle
// inequality it can be exceeded, so "exact within the probed hoods" becomes
// a recall guarantee at boundConfidence. Query.Exact turns it off (every row
// survives the bound pass) for callers that want the certainty back.
const (
	bitsMagic = "DQB1"
	// rowsMagic is a DQB1 part without its centroid (Quantizer.MarshalRows):
	// a store that files many code blocks under one cluster keeps the
	// centroid once and hands it back to UnmarshalRowsBorrowed.
	rowsMagic = "DQR1"
	// bitsHeader is magic(4) version(2) metric(1) qbits(1) dims(4) rows(4)
	// seed(8).
	bitsHeader   = 24
	bitsVersion  = 1
	queryBits    = 4
	queryLevels  = 1<<queryBits - 1
	metricRaw    = 0 // rows are the vectors themselves (l2)
	metricUnit   = 1 // rows are unit vectors (cosine)
	zeroRowAlign = -1
)

// newBitScorer sets the four planes unrolled: compile only at four bits.
var _ [queryBits - 4]struct{}
var _ [4 - queryBits]struct{}

// boundSigmas is eps0: how many standard deviations of estimator error the
// conservative bound carries. The error is a projection of a fixed vector
// onto a direction the random rotation makes uniform, so it concentrates
// like a Gaussian and eps0 reads straight off the normal tail: the bound is
// one-sided (only an OVER-estimate of the distance can wrongly skip a row),
// so 2.5 sigmas is a per-row failure probability of 6.2e-3.
//
// The paper uses 1.9 and calls it "nearly perfect confidence"; that is 2.9%
// one-sided. Measured on BIGANN 10M at 16, 32 and 98 probes, 2.5 reads
// 56-58% fewer rows in the bound-pruned pass than 3.2 for recall identical
// at three decimals and equal to reading every scored row; 1.9 starts to
// cost (0.001). A per-row failure is only a top-k flip when the row is a
// true neighbour within one bound width of the cutoff, and the exact
// shortlist has already pushed the cutoff past almost all of those (the
// margin argument of arXiv 2609.09854). Query.Sigmas overrides it per
// query; TestBitBoundCoversAtItsConfidence measures the real coverage.
const boundSigmas = 2.5

// Code is the 1-bit payload of a Quantizer. Rows are indexed in the order
// the vectors were given to Quantize, which is the order a caller labels
// scores with.
type Code struct {
	// Words is rows x ceil(dims/64) little-endian bit rows: bit j of row i
	// is word[i*w + j/64] >> (j%64).
	Words []uint64
	// Centroid is the cluster's own mean, in the ORIGINAL frame, subtracted
	// before rotation. It is the codec's, not the index's: a cluster's
	// members drift from the centroid they were filed under, and the
	// estimator's accuracy is a function of how short the residuals are,
	// so the codec carries the centroid it actually encoded against.
	Centroid []float32
	// Norms is ||o - c|| per row, and Aligns is <x_b, x> per row: the two
	// scalars the estimator needs. Aligns is zeroRowAlign for a row whose
	// vector has no direction at all (a zero vector in a cosine column),
	// which scores 0 the way CosineSim scores it.
	Norms  []float32
	Aligns []float32
	// Seed is the rotation the bits live in (rotate.go). Stored rather than
	// derived so a reader never has to agree with the writer about how a
	// seed is computed.
	Seed uint64
	// Unit says the rows are unit vectors: cosine columns quantize the
	// residual of the NORMALIZED vector, so the estimator's squared
	// distance maps to cosine as 1 - d^2/2.
	Unit bool
	// borrowed is the encoded column UnmarshalBinaryBorrowed aliased, read
	// in place; every other column keeps the typed slices above.
	borrowed []byte
	rows     int
	dims     int
	// cOff is the centroid's bytes in borrowed: 4*dims for DQB1, 0 for
	// DQR1, whose centroid is the Centroid slice the caller handed in.
	cOff int
	// fast is the rows packed for the batched scan (PackFastScan); nil
	// until a caller packs the column.
	fast *fastLayout
}

// fastLayout is a column's bits as simd.FastScan reads them: blocks of 32
// rows, one nibble (four dimensions) per row per group, and each row's
// popcount, which the scan's tables do not carry.
type fastLayout struct {
	groups, blocks int
	nibs           []byte
	ones           []uint16
	// norms and aligns are the column's, decoded once: a borrowed column
	// reads them out of its bytes per row.
	norms, aligns []float32
	// xbc is <x_b, Rc> per row, the column's rotated centroid against the
	// row's code (PackFastScanFrame), and rcc ||Rc||^2: what a global query
	// (Query.Global) subtracts instead of quantizing its residual per
	// column. nil until packed with a frame.
	xbc []float32
	rc  []float32
	rcc float64
	// ownCentroid: packing copied the centroid out of the borrowed bytes.
	// Otherwise it is the caller's (UnmarshalRowsBorrowed), shared by every
	// block of a cluster, and retained does not charge it per column.
	ownCentroid bool
}

// fastGroups is the scan's group count at dims: a nibble per four
// dimensions, even, or 0 when the column is too wide to scan batched.
func fastGroups(dims int) int {
	g := (dims + 3) / 4
	g += g % 2
	if g > simd.FastScanMaxGroups {
		return 0
	}
	return g
}

// PackFastScanFrame is PackFastScan, plus what a global query needs
// (Scorer.BindGlobal): rc is the column's centroid rotated into its codes'
// frame (RotateCentroid), and each row's <x_b, rc> is kept. The column
// keeps rc.
func (c *Quantizer) PackFastScanFrame(rc []float32) {
	b := c.Code
	if b.fast != nil && b.fast.xbc != nil {
		return
	}
	d := b.width()
	if len(rc) != d || fastGroups(d) == 0 {
		c.PackFastScan()
		return
	}
	rows, words := b.Rows(), bitWords(d)
	var sum float64
	for _, v := range rc {
		sum += float64(v)
	}
	scale := 1 / math.Sqrt(float64(d))
	xbc := make([]float32, rows)
	for r := range rows {
		// <x_b, rc> = (2 Σ_{b_j=1} rc_j - Σ rc_j) / sqrt(D)
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
	var rcc float64
	for _, v := range rc {
		rcc += float64(v) * float64(v)
	}
	c.PackFastScan()
	if b.fast == nil {
		return
	}
	b.fast.xbc, b.fast.rc, b.fast.rcc = xbc, rc, rcc
}

// PackFastScan lays the column's rows out for Scorer.ScoreAll's batched
// scan, which then scores 32 rows per table lookup instead of a row at a
// time: about 1 ns a row against 28 for the popcount product alone, at
// 128 dimensions. The packed rows replace the column's bits: it stops
// aliasing a borrowed input (UnmarshalBinaryBorrowed, UnmarshalRowsBorrowed)
// and holds about what it did, so a cache of columns holds as many. Kept
// beside the bits, they cost as much memory again, and at 10M rows in a
// 1 GiB cache that tripled a warm query (753 loads a query against 37).
// Every method still works on a packed column. It writes the column, so
// call it before the column is shared; packing twice is a no-op, and a
// column too wide for the scan stays unpacked.
func (c *Quantizer) PackFastScan() {
	b := c.Code
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
		// The centroid lives in the borrowed bytes: keep a copy. A
		// centroid the caller handed in (cOff 0) is already b.Centroid.
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

func bitWords(dims int) int { return (dims + 63) / 64 }

// width is the column's dims, whichever representation holds it.
func (b *Code) width() int {
	if b.borrowed != nil {
		return b.dims
	}
	return len(b.Centroid)
}

// Rows is how many code rows the payload holds.
func (b *Code) Rows() int {
	if b.borrowed != nil {
		return b.rows
	}
	return len(b.Norms)
}

func (b *Code) centroid(i int) float32 {
	if b.borrowed != nil && b.cOff > 0 {
		return math.Float32frombits(binary.LittleEndian.Uint32(b.borrowed[bitsHeader+4*i:]))
	}
	return b.Centroid[i]
}

func (b *Code) norm(i int) float32 {
	if b.borrowed != nil {
		return math.Float32frombits(binary.LittleEndian.Uint32(b.borrowed[bitsHeader+b.cOff+4*i:]))
	}
	return b.Norms[i]
}

func (b *Code) align(i int) float32 {
	if b.borrowed != nil {
		return math.Float32frombits(binary.LittleEndian.Uint32(b.borrowed[bitsHeader+b.cOff+4*b.rows+4*i:]))
	}
	return b.Aligns[i]
}

func (b *Code) word(i int) uint64 {
	if b.borrowed != nil {
		return binary.LittleEndian.Uint64(b.borrowed[bitsHeader+b.cOff+8*b.rows+8*i:])
	}
	if b.Words == nil && b.fast != nil {
		return b.fast.word(i, bitWords(b.width()))
	}
	return b.Words[i]
}

// word is word i of a packed column (words per row): its 16 nibbles,
// reassembled. Cold paths only (marshal, Select, AppendRows); scoring
// reads the nibbles.
func (f *fastLayout) word(i, words int) uint64 {
	r, w := i/words, i%words
	k, j, shift := r/32, r%32, uint(0)
	if j >= 16 {
		j, shift = j-16, 4
	}
	var x uint64
	for n := range 16 {
		g := w*16 + n
		if g >= f.groups {
			break
		}
		x |= uint64(f.nibs[(k*f.groups+g)*16+j]>>shift&0xF) << (4 * n)
	}
	return x
}

// quantizeBits encodes vectors as 1-bit residuals to their own mean.
// metric decides whether the rows are normalized first; seed is the
// rotation. An empty corpus still produces a valid, zero-row payload,
// because a hood may legitimately hold only tombstones.
func quantizeBits(vectors [][]float32, metric ivfq.Metric, rot *Rotation) (Quantizer, error) {
	dims, seed := rot.dims, rot.seed
	if metric != ivfq.L2 && metric != ivfq.Cosine {
		return Quantizer{}, fmt.Errorf("rabitq: metric %s has no 1-bit codec", metric)
	}
	if dims <= 0 {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes need a positive width, got %d", dims)
	}
	if seed == 0 {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes need a nonzero rotation seed")
	}
	for i, v := range vectors {
		if len(v) != dims {
			return Quantizer{}, fmt.Errorf("quant: row %d is %d-wide, the field is %d", i, len(v), dims)
		}
	}
	unit := metric == ivfq.Cosine
	rows := len(vectors)
	b := &Code{
		Words:    make([]uint64, rows*bitWords(dims)),
		Centroid: make([]float32, dims),
		Norms:    make([]float32, rows),
		Aligns:   make([]float32, rows),
		Seed:     seed,
		Unit:     unit,
	}
	// The rows the centroid is the mean of, in the frame they are encoded
	// in: unit vectors for cosine, the vectors themselves for l2. A zero
	// vector has no unit form, so it is not a member of the mean and not a
	// row the estimator speaks for.
	work := make([][]float32, rows)
	sum := make([]float64, dims)
	members := 0
	for i, v := range vectors {
		row := workRow(v, unit)
		if row == nil {
			b.Aligns[i] = zeroRowAlign
			continue
		}
		work[i] = row
		for j, x := range row {
			sum[j] += float64(x)
		}
		members++
	}
	if members > 0 {
		for j := range b.Centroid {
			b.Centroid[j] = float32(sum[j] / float64(members))
		}
	}
	b.fillRows(work, rot)
	return Quantizer{Dims: dims, Code: b}, nil
}

func b2u(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

// fillRows is fillRow over every non-nil row of work, simd.FillBlock rows at a
// time: the residuals are laid out dimension-major, so each butterfly pair
// is loaded once per block and rotates a contiguous run of simd.FillBlock
// values instead of two scattered ones per row, and the per-row float64
// sums (the norm, the alignment's |x|) run as simd.FillBlock independent chains
// instead of one latency-bound chain per row. Every value goes through the
// same float32 operations in the same order as Rotation.Apply, and every
// row's sum adds its terms in the same order as fillRow, so the codes are
// bit-identical to fillRow's (TestFillRowsMatchesFillRow).
func (b *Code) fillRows(work [][]float32, rot *Rotation) {
	dims := len(b.Centroid)
	w := bitWords(dims)
	block := make([]float32, dims*simd.FillBlock) // block[j*simd.FillBlock+r]: dim j of the block's row r
	scale := math.Sqrt(float64(dims))
	var rows [simd.FillBlock]int
	for start := 0; start < len(work); {
		n := 0
		for ; start < len(work) && n < simd.FillBlock; start++ {
			row := work[start]
			if row == nil {
				continue // zero row in a cosine column
			}
			for j, c := range b.Centroid {
				block[j*simd.FillBlock+n] = row[j] - c
			}
			rows[n] = start
			n++
		}
		var norms, absSums [simd.FillBlock]float64
		for j := range dims {
			col := block[j*simd.FillBlock:][:simd.FillBlock]
			for r := range n {
				x := float64(col[r])
				norms[r] += x * x
			}
		}
		for r, i := range rows[:n] {
			norms[r] = math.Sqrt(norms[r])
			b.Norms[i] = float32(norms[r]) // 0: the row IS the centroid, no bits, alignment 0
		}
		rot.applyBlock(block) // rows past n are stale and ignored
		// Branch-free: a residual's signs are coin flips, so a branch on
		// each mispredicts half the time. |x| accumulates exactly what
		// fillRow's +x / -x does.
		var words [simd.FillBlock]uint64
		for j := range dims {
			col := block[j*simd.FillBlock:][:simd.FillBlock]
			shift := uint(j % 64)
			for r := range n {
				x := col[r]
				words[r] |= b2u(x >= 0) << shift
				absSums[r] += math.Abs(float64(x))
			}
			if shift == 63 || j == dims-1 {
				for r, i := range rows[:n] {
					if b.Norms[i] != 0 {
						b.Words[i*w+j/64] = words[r]
					}
					words[r] = 0
				}
			}
		}
		for r, i := range rows[:n] {
			if b.Norms[i] == 0 {
				continue
			}
			// <x_b, x> with x_b = (2bit-1)/sqrt(D) and x = residual/norm.
			align := absSums[r] / (norms[r] * scale)
			b.Aligns[i] = float32(min(1, align))
		}
	}
}

// workRow keeps an L2 row and copies only when normalizing cosine. nil
// means a zero vector has no unit form — a cosine row that carries
// zeroRowAlign instead of bits.
func workRow(v []float32, unit bool) []float32 {
	if !unit {
		return v
	}
	row := make([]float32, len(v))
	copy(row, v)
	var n float64
	for _, x := range row {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return nil
	}
	inv := float32(1 / math.Sqrt(n))
	for j := range row {
		row[j] *= inv
	}
	return row
}

// fillRow encodes one working row's residual to the column's centroid into
// row i: the stored norm, the rotated residual's sign bits, and the
// alignment scalar. row is already at working precision (unit-normalized
// for a cosine column) and non-nil. A norm of 0 — the row IS the centroid,
// or is within a float32 ulp of it — stores no bits and an alignment of 0,
// which is exactly the no-norm-no-alignment agreement the reader checks.
// The test is on the STORED norm, because that is what the reader sees.
func (b *Code) fillRow(i int, row []float32, rot *Rotation, scratch []float32) {
	dims := len(b.Centroid)
	w := bitWords(dims)
	var n float64
	for j := range scratch {
		scratch[j] = row[j] - b.Centroid[j]
		n += float64(scratch[j]) * float64(scratch[j])
	}
	n = math.Sqrt(n)
	b.Norms[i] = float32(n)
	if b.Norms[i] == 0 {
		return
	}
	rot.Apply(scratch)
	var absSum float64
	for j, x := range scratch {
		if x >= 0 {
			b.Words[i*w+j/64] |= 1 << uint(j%64)
			absSum += float64(x)
		} else {
			absSum -= float64(x)
		}
	}
	// <x_b, x> with x_b = (2bit-1)/sqrt(D) and x = scratch/n.
	align := absSum / (n * math.Sqrt(float64(dims)))
	b.Aligns[i] = float32(min(1, align))
}

func (b *Code) marshal(dims int, withCentroid bool) ([]byte, error) {
	if b.borrowed != nil && withCentroid == (b.cOff > 0) {
		return bytes.Clone(b.borrowed), nil
	}
	rows, w := b.Rows(), bitWords(dims)
	if b.borrowed == nil && (len(b.Aligns) != rows || len(b.Centroid) != dims || b.fast == nil && len(b.Words) != rows*w) {
		return nil, fmt.Errorf("quant: inconsistent 1-bit codes")
	}
	var out bytes.Buffer
	out.Grow(bitsHeader + 4*dims + 8*rows + 8*rows*w)
	magic := rowsMagic
	if withCentroid {
		magic = bitsMagic
	}
	out.WriteString(magic)
	var scratch [8]byte
	put16 := func(v uint16) { binary.LittleEndian.PutUint16(scratch[:2], v); out.Write(scratch[:2]) }
	put32 := func(v uint32) { binary.LittleEndian.PutUint32(scratch[:4], v); out.Write(scratch[:4]) }
	put64 := func(v uint64) { binary.LittleEndian.PutUint64(scratch[:8], v); out.Write(scratch[:8]) }
	put16(bitsVersion)
	metric := byte(metricRaw)
	if b.Unit {
		metric = metricUnit
	}
	out.WriteByte(metric)
	out.WriteByte(queryBits)
	put32(uint32(dims))
	put32(uint32(rows))
	put64(b.Seed)
	if withCentroid {
		for i := range dims {
			put32(math.Float32bits(b.centroid(i)))
		}
	}
	for i := range rows {
		put32(math.Float32bits(b.norm(i)))
	}
	for i := range rows {
		put32(math.Float32bits(b.align(i)))
	}
	for i := range rows * w {
		put64(b.word(i))
	}
	return out.Bytes(), nil
}

// unmarshalBits decodes and validates a 1-bit codes part. Everything a row's
// score depends on is checked here, before any of it prunes a rerank: the
// exact length the header implies, finite scalars, a nonnegative residual
// norm, an alignment inside the range the encoder can produce (it is
// <x_b, x> for a unit x, so it cannot be below 1/sqrt(D) nor above 1), the
// agreement between "no residual" and "no alignment", and zero padding in
// the bits past the last dimension.
func unmarshalBits(data []byte) (Quantizer, error) { return decodeBits(data, true, false, nil) }

func unmarshalBitsBorrowed(data []byte) (Quantizer, error) { return decodeBits(data, true, true, nil) }

// decodeBits decodes a DQB1 part (withCentroid) or a DQR1 part, whose
// centroid is the one given.
func decodeBits(data []byte, withCentroid, borrow bool, centroid []float32) (Quantizer, error) {
	magic := rowsMagic
	if withCentroid {
		magic = bitsMagic
	}
	if len(data) < bitsHeader || string(data[:4]) != magic {
		return Quantizer{}, fmt.Errorf("quant: invalid 1-bit codes header")
	}
	if v := binary.LittleEndian.Uint16(data[4:]); v != bitsVersion {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes version %d, this reader speaks %d", v, bitsVersion)
	}
	metric, qbits := data[6], data[7]
	if metric != metricRaw && metric != metricUnit {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes metric byte %d", metric)
	}
	if qbits != queryBits {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes want %d query bits, this reader has %d", qbits, queryBits)
	}
	dims, rows := int(binary.LittleEndian.Uint32(data[8:])), int(binary.LittleEndian.Uint32(data[12:]))
	seed := binary.LittleEndian.Uint64(data[16:])
	if dims <= 0 || rows < 0 || seed == 0 {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes shape %dx%d seed %d", rows, dims, seed)
	}
	cOff := 0
	if withCentroid {
		cOff = 4 * dims
	} else if len(centroid) != dims {
		return Quantizer{}, fmt.Errorf("quant: a %d-wide centroid for %d-wide codes", len(centroid), dims)
	}
	w := bitWords(dims)
	want := uint64(bitsHeader) + uint64(cOff) + 8*uint64(rows) + 8*uint64(rows)*uint64(w)
	if want != uint64(len(data)) {
		return Quantizer{}, fmt.Errorf("quant: 1-bit codes are %d bytes, a %dx%d part is %d", len(data), rows, dims, want)
	}
	b := &Code{Seed: seed, Unit: metric == metricUnit, rows: rows, dims: dims, cOff: cOff}
	if !withCentroid {
		b.Centroid = centroid
	}
	if borrow {
		b.borrowed = data
	} else {
		b.Words = make([]uint64, rows*w)
		b.Norms = make([]float32, rows)
		b.Aligns = make([]float32, rows)
		if withCentroid {
			b.Centroid = make([]float32, dims)
			for i := range b.Centroid {
				b.Centroid[i] = b.centroidFrom(data, i)
			}
		}
		for i := range b.Norms {
			b.Norms[i] = b.normFrom(data, i)
			b.Aligns[i] = b.alignFrom(data, i)
		}
		for i := range b.Words {
			b.Words[i] = b.wordFrom(data, i)
		}
	}
	for i := 0; i < dims; i++ {
		x := b.centroid(i)
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return Quantizer{}, fmt.Errorf("quant: centroid dimension %d is %v", i, x)
		}
	}
	floor := 0.999 / math.Sqrt(float64(dims))
	for i := 0; i < rows; i++ {
		n, a := b.norm(i), b.align(i)
		if !finiteNonnegative(n) {
			return Quantizer{}, fmt.Errorf("quant: row %d residual norm %v", i, n)
		}
		if a == zeroRowAlign {
			if !b.Unit || n != 0 {
				return Quantizer{}, fmt.Errorf("quant: row %d is marked directionless but carries norm %v", i, n)
			}
			continue
		}
		if math.IsNaN(float64(a)) || a < 0 || a > 1 {
			return Quantizer{}, fmt.Errorf("quant: row %d alignment %v", i, a)
		}
		if (n == 0) != (a == 0) {
			return Quantizer{}, fmt.Errorf("quant: row %d residual norm %v and alignment %v disagree", i, n, a)
		}
		if n != 0 && float64(a) < floor {
			return Quantizer{}, fmt.Errorf("quant: row %d alignment %v is below the %v a %d-wide sign code can produce", i, a, floor, dims)
		}
		if n == 0 {
			for j := 0; j < w; j++ {
				if b.word(i*w+j) != 0 {
					return Quantizer{}, fmt.Errorf("quant: row %d has no residual but carries bits", i)
				}
			}
		}
		if pad := uint(w*64 - dims); pad > 0 && b.word((i+1)*w-1)>>(64-pad) != 0 {
			return Quantizer{}, fmt.Errorf("quant: row %d sets bits past dimension %d", i, dims)
		}
	}
	if !borrow {
		b.rows, b.dims, b.cOff = 0, 0, 0 // owned Code keeps its historical shape
	}
	return Quantizer{Dims: dims, Code: b}, nil
}

// sameFrame reports whether o's rows were encoded exactly as b's: same
// width, seed, metric frame and centroid, so a query bound to b scores o.
func (b *Code) sameFrame(o *Code) bool {
	if b == o {
		return true
	}
	if b == nil || o == nil || b.Seed != o.Seed || b.Unit != o.Unit || b.width() != o.width() {
		return false
	}
	if len(b.Centroid) > 0 && len(o.Centroid) == len(b.Centroid) && &b.Centroid[0] == &o.Centroid[0] {
		return true // one centroid slice: codes decoded against a shared centroid
	}
	for i := 0; i < b.width(); i++ {
		if b.centroid(i) != o.centroid(i) {
			return false
		}
	}
	return true
}

func (b *Code) centroidFrom(data []byte, i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data[bitsHeader+4*i:]))
}
func (b *Code) normFrom(data []byte, i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data[bitsHeader+b.cOff+4*i:]))
}
func (b *Code) alignFrom(data []byte, i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data[bitsHeader+b.cOff+4*b.rows+4*i:]))
}
func (b *Code) wordFrom(data []byte, i int) uint64 {
	return binary.LittleEndian.Uint64(data[bitsHeader+b.cOff+8*b.rows+8*i:])
}

// bitScorer is one hood's bits bound to one query: the query's residual to
// THIS hood's centroid, rotated into the bits' frame, normalized, and
// quantized to 4 bits per dimension as four bit planes the row loop
// popcounts against (planes: plane p is planes[p*words:(p+1)*words]).
type bitScorer struct {
	b      *Code
	dims   int
	words  int
	planes []uint64
	// lut is simd.FastScan's tables for this query: entry 16g+n is
	// Σ code_j over the set bits of nibble n of group g. sums is the
	// scan's output, reused across the columns this scorer is rebound to.
	lut  []byte
	sums []uint16

	qNorm  float64 // ||q - c||
	delta  float64 // query quantization step
	low    float64 // v_l, the bottom of the query's range
	sumQ   float64 // sum of the integer query codes
	scale  float64 // 1/sqrt(dims)
	resid2 float64 // ||u~ - u||^2, the query's own rounding residual
	varQ   float64 // resid2/dims, the query term of the estimator's variance
	freeD  float64 // max(dims-1, 1)

	// g is the global query a BindGlobal scorer reads, with inv
	// 1/||q - c||; nil for a scorer bound per column.
	g   *globalQuery
	inv float64

	dead   bool // a zero cosine query: everything scores 0
	sigmas float64
	exact  bool
	l2     bool
}

func (c *Quantizer) newBitScorer(q Query, rc []float32) *bitScorer {
	s := new(bitScorer)
	c.bind(s, q, rc)
	return s
}

// bindGlobal sets s to the global query q against c (BindGlobal checked
// both): the query's quantization is q's, and the column's part is its
// distance to c's centroid.
func (c *Quantizer) bindGlobal(s *bitScorer, q Query) {
	b := c.Code
	d := c.Dims
	planes, lut, sums := s.planes, s.lut, s.sums
	*s = bitScorer{b: b, dims: d, words: bitWords(d), scale: 1 / math.Sqrt(float64(d)),
		freeD: math.Max(float64(d-1), 1), sigmas: boundSigmas, exact: q.Exact,
		l2: q.Metric == ivfq.L2, planes: planes[:0], lut: lut[:0], sums: sums}
	if q.Sigmas > 0 {
		s.sigmas = q.Sigmas
	}
	if b.Unit && q.invNorm == 0 {
		s.dead = true
		return
	}
	g := q.global
	f := b.fast
	n2 := g.qq + f.rcc - 2*float64(ivfq.Dot(q.rotated, f.rc))
	s.qNorm = math.Sqrt(max(0, n2))
	if s.qNorm == 0 {
		return
	}
	s.g, s.inv = g, 1/s.qNorm
	s.low, s.delta, s.sumQ = g.low, g.delta, g.sumQ
	s.resid2 = g.resid2 * s.inv * s.inv
	s.varQ = s.resid2 / float64(d)
}

// bind sets s to q against c, reusing s's buffers.
func (c *Quantizer) bind(s *bitScorer, q Query, rc []float32) {
	b := c.Code
	d := c.Dims
	planes, lut, sums := s.planes, s.lut, s.sums
	*s = bitScorer{b: b, dims: d, words: bitWords(d), scale: 1 / math.Sqrt(float64(d)),
		freeD: math.Max(float64(d-1), 1), sigmas: boundSigmas, exact: q.Exact,
		l2: q.Metric == ivfq.L2, sums: sums}
	if q.Sigmas > 0 {
		s.sigmas = q.Sigmas
	}
	// Three ways the query cannot be scored against these bits at all: a
	// metric this codec does not know, a query narrower than the codes, and
	// a query whose metric disagrees with the frame the rows were encoded
	// in (cosine rows are unit vectors, l2 rows are not; a field cannot
	// change its metric, so this is corruption or a bug rather than a
	// case). Each one scores nothing and, crucially, BOUNDS nothing: the
	// bound pass then reads every row and the answer is still right, just
	// slow. A bound of zero here would silently prune the whole index.
	if (!s.l2 && q.Metric != ivfq.Cosine) || len(q.Vector) < d || b.Unit != (q.Metric == ivfq.Cosine) ||
		q.Rotation == nil || q.Rotation.seed != b.Seed || q.Rotation.dims != d {
		s.dead, s.exact = true, true
		return
	}
	if b.Unit && q.invNorm == 0 {
		s.dead = true
		return
	}
	var residBuf [512]float32 // on the stack: resid does not escape
	resid := residBuf[:0]
	if d <= len(residBuf) {
		resid = residBuf[:d]
	} else {
		resid = make([]float32, d)
	}
	var n float64
	if rc != nil && q.rotated != nil {
		// Rotated: R(q-c) = Rq - Rc, no rotation per column.
		for j := range resid {
			resid[j] = q.rotated[j] - rc[j]
			n += float64(resid[j]) * float64(resid[j])
		}
	} else {
		copy(resid, q.Vector[:d])
		if b.Unit {
			inv := float32(q.invNorm)
			for j := range resid {
				resid[j] *= inv
			}
		}
		for j := range resid {
			resid[j] -= b.centroid(j)
			n += float64(resid[j]) * float64(resid[j])
		}
		q.Rotation.Apply(resid)
	}
	s.qNorm = math.Sqrt(n)
	if s.qNorm == 0 {
		return // the query IS the centroid: every row's distance is its own norm
	}
	// The unit query residual, then 4-bit uniform scalar quantization over
	// its own range. Rounding is to NEAREST, not the paper's randomized
	// rounding: a query must answer the same twice, and the bias that costs
	// is carried explicitly in the bound instead, measured per query as the
	// rounding residual's norm rather than bounded worst-case (a worst-case
	// query term is sqrt(D)*delta/2, three times the data term at 128 dims,
	// and it would swamp a bound whose whole job is to prune).
	// Once per probed hood, so it is a hood's fixed cost: the builtin
	// min and max inline where math.Min and math.Max are calls, the sums
	// stay in registers, and the planes are set without a branch per bit.
	// The codes are the same, bit for bit.
	inv := 1 / s.qNorm
	lo, hi := math.Inf(1), math.Inf(-1)
	for j := range resid {
		x := float64(resid[j]) * inv
		lo, hi = min(lo, x), max(hi, x)
	}
	delta := (hi - lo) / queryLevels
	s.low, s.delta = lo, delta
	if n := queryBits * s.words; cap(planes) >= n {
		s.planes = planes[:n]
		clear(s.planes)
	} else {
		s.planes = make([]uint64, n)
	}
	p0, p1, p2, p3 := s.planes[:s.words], s.planes[s.words:2*s.words], s.planes[2*s.words:3*s.words], s.planes[3*s.words:]
	var sumQ, resid2 float64
	// The scan's tables take each code straight into its single-bit
	// entry (16g + 1<<i for dimension 4g+i); the sums fill the rest after.
	if groups := fastGroups(d); groups > 0 {
		if n := 16 * groups; cap(lut) >= n {
			s.lut = lut[:n]
			clear(s.lut)
		} else {
			s.lut = make([]byte, n)
		}
	}
	lut = s.lut
	for j := range resid {
		x := float64(resid[j]) * inv
		code := 0
		if delta > 0 {
			// math.Round, exactly, for y >= 0 (x >= lo): Trunc is one
			// instruction where Round is a bit-twiddling call, and
			// y - Trunc(y) is exact.
			y := (x - lo) / delta
			t := math.Trunc(y)
			if y-t >= 0.5 {
				t++
			}
			code = int(t)
			code = min(queryLevels, max(0, code))
		}
		sumQ += float64(code)
		e := lo + delta*float64(code) - x
		resid2 += e * e
		w, sh := j/64, uint(j%64)
		c := uint64(code)
		p0[w] |= (c & 1) << sh
		p1[w] |= (c >> 1 & 1) << sh
		p2[w] |= (c >> 2 & 1) << sh
		p3[w] |= (c >> 3 & 1) << sh
		if lut != nil {
			lut[16*(j/4)+1<<(j%4)] = byte(code)
		}
	}
	s.sumQ, s.resid2 = sumQ, resid2
	for g := 0; g < len(lut); g += 16 {
		t := lut[g : g+16 : g+16]
		t[3] = t[1] + t[2]
		t[5], t[6], t[7] = t[4]+t[1], t[4]+t[2], t[4]+t[3]
		for n := 9; n < 16; n++ {
			t[n] = t[8] + t[n-8]
		}
	}
	s.varQ = s.resid2 / float64(d)
}

// rowIP is <x_b, u~>: the four bit-plane popcounts, plus the row's own
// popcount for the v_l term, in one pass over the row's words
// (bitproduct.go; the AVX2 kernel on amd64). s.b is read here, not at
// binding: Quantizer.Scorer rebinds a scorer to another hood's codes in
// the same frame, and that hood has its own row count and representation.
func (s *bitScorer) rowIP(row int) float64 {
	if s.g != nil {
		return s.globalIP(row)
	}
	off := row * s.words
	var ones, weighted uint64
	switch b := s.b; {
	case b.borrowed != nil:
		ones, weighted = simd.BitProductBytes(b.borrowed[bitsHeader+b.cOff+8*b.rows+8*off:], s.planes, s.words)
	case b.Words != nil:
		ones, weighted = simd.BitProductWords(b.Words[off:off+s.words], s.planes)
	default: // packed: Σ_j b_j code_j is the row's table entries summed
		f := b.fast
		k, i, shift := row/32, row%32, uint(0)
		if i >= 16 {
			i, shift = i-16, 4
		}
		ones = uint64(f.ones[row])
		if len(s.lut) == 16*f.groups {
			for g := range f.groups {
				weighted += uint64(s.lut[16*g+int(f.nibs[(k*f.groups+g)*16+i]>>shift&0xF)])
			}
		} else {
			words := make([]uint64, s.words)
			for w := range words {
				words[w] = f.word(off+w, s.words)
			}
			ones, weighted = simd.BitProductWords(words, s.planes)
		}
	}
	return s.ipOf(ones, weighted)
}

// globalIP is rowIP for a global query: <x_b, Rq~> from the row's
// nibbles through the query's two tables, less the row's <x_b, Rc>, over
// ||q - c||.
func (s *bitScorer) globalIP(row int) float64 {
	f := s.b.fast
	k, i, shift := row/32, row%32, uint(0)
	if i >= 16 {
		i, shift = i-16, 4
	}
	var lo, hi uint64
	for g := range f.groups {
		n := int(f.nibs[(k*f.groups+g)*16+i] >> shift & 0xF)
		lo += uint64(s.g.lutLo[16*g+n])
		hi += uint64(s.g.lutHi[16*g+n])
	}
	return (s.ipOf(uint64(f.ones[row]), lo+16*hi) - float64(f.xbc[row])) * s.inv
}

// ipOf is rowIP from a row's popcount and its weighted popcount
// Σ_j b_j code_j, however they were counted.
func (s *bitScorer) ipOf(ones, weighted uint64) float64 {
	// sum_j (2b_j - 1) u~_j / sqrt(D), with u~_j = v_l + delta*code_j.
	dot := 2*(s.delta*float64(weighted)+s.low*float64(ones)) - (s.delta*s.sumQ + float64(s.dims)*s.low)
	return dot * s.scale
}

// scoreAndBound is the estimator and its error bound for one row.
func (s *bitScorer) scoreAndBound(row int) (float32, float32) { return s.estimate(row, false, 0, true) }

// score is scoreAndBound's estimate alone, without the bound's arithmetic.
func (s *bitScorer) score(row int) float32 {
	score, _ := s.estimate(row, false, 0, false)
	return score
}

// estimate is scoreAndBound, with the row's rowIP already counted when
// known (the batched scan) and the bound skipped unless bound: the one
// arithmetic every path scores through, so they agree bit for bit.
func (s *bitScorer) estimate(row int, known bool, rip float64, bound bool) (float32, float32) {
	if s.dead {
		if s.exact {
			return 0, float32(math.Inf(1))
		}
		return 0, 0 // a zero cosine query: every score IS 0, and 0 bounds it
	}
	align := float64(s.b.align(row))
	if align == zeroRowAlign {
		return 0, 0 // a zero vector's cosine is 0, exactly
	}
	norm := float64(s.b.norm(row))
	dist2, slack := norm*norm+s.qNorm*s.qNorm, 0.0
	if norm > 0 && s.qNorm > 0 && align > 0 {
		if !known {
			rip = s.rowIP(row)
		}
		ip := rip / align
		dist2 -= 2 * norm * s.qNorm * ip
		if !bound {
			gap := norm - s.qNorm
			return float32(scoreOf(s.l2, max(dist2, gap*gap))), 0
		}
		// The paper's error term, sqrt(1-align^2)/align/sqrt(D-1), plus
		// this query's own rounding residual, added in quadrature because
		// the two projections are onto independent directions of the same
		// random frame.
		//
		// The data term is a projection onto the part of the query residual
		// ORTHOGONAL to this row's, so it carries a factor
		// ||u_perp|| = sqrt(1 - <x,u>^2) the paper bounds by 1. Bounding it
		// instead by the smallest <x,u> the estimate itself allows tightens
		// exactly the rows that matter — the near ones, whose <x,u> is
		// large — and those are the rows a cutoff is deciding about. Two
		// steps, because the first bound is what makes the second sound.
		sigma := math.Sqrt((1-align*align)/s.freeD+s.varQ) / align
		if near := math.Abs(ip) - s.sigmas*sigma; near > 0 {
			perp2 := max(0, 1-near*near)
			sigma = math.Sqrt((1-align*align)*perp2/s.freeD+s.varQ) / align
		}
		slack = 2 * norm * s.qNorm * s.sigmas * sigma
	}
	// The triangle inequality holds whatever the estimate says: two vectors
	// whose residuals differ in length by g are at least g apart, and two
	// on a common sphere are at most 2r apart.
	gap := norm - s.qNorm
	dist2 = max(dist2, gap*gap)
	if !bound {
		return float32(scoreOf(s.l2, dist2)), 0
	}
	if s.exact {
		return float32(scoreOf(s.l2, dist2)), float32(math.Inf(1))
	}
	lower := max(dist2-slack, gap*gap)
	return float32(scoreOf(s.l2, dist2)), nextUp(float32(scoreOf(s.l2, lower)))
}

// scoreAll is score (bounds nil) or scoreAndBound for every row, through
// the batched scan when the column is packed and the query has tables.
func (s *bitScorer) scoreAll(scores, bounds []float32) {
	rows := s.b.Rows()
	f := s.b.fast
	if s.g != nil && !s.dead && s.qNorm > 0 {
		s.scoreAllGlobal(f, scores, bounds)
		return
	}
	if f == nil || s.dead || len(s.lut) != 16*f.groups {
		for r := range rows {
			if bounds != nil {
				scores[r], bounds[r] = s.scoreAndBound(r)
			} else {
				scores[r] = s.score(r)
			}
		}
		return
	}
	if cap(s.sums) < 32*f.blocks {
		s.sums = make([]uint16, 32*f.blocks)
	}
	sums := s.sums[:32*f.blocks]
	simd.FastScan(f.nibs, s.lut, f.groups, f.blocks, sums)
	if bounds == nil {
		s.scoreFast(f, sums, scores)
		return
	}
	for r := range rows {
		ip := s.ipOf(uint64(f.ones[r]), uint64(sums[r]))
		if bounds != nil {
			scores[r], bounds[r] = s.estimate(r, true, ip, true)
		} else {
			scores[r], _ = s.estimate(r, true, ip, false)
		}
	}
}

// scoreAllGlobal is scoreAll for a global query: one scan over the rows'
// nibbles through the query's two tables (its codes' low and high halves)
// gives each row's Σ b_j code_j.
func (s *bitScorer) scoreAllGlobal(f *fastLayout, scores, bounds []float32) {
	n := 32 * f.blocks
	if cap(s.sums) < 2*n {
		s.sums = make([]uint16, 2*n)
	}
	lo, hi := s.sums[:n], s.sums[n:2*n]
	simd.FastScan2(f.nibs, s.g.lutLo, s.g.lutHi, f.groups, f.blocks, lo, hi)
	if bounds == nil {
		s.scoreFastGlobal(f, lo, hi, scores)
		return
	}
	for r := range scores {
		w := uint64(lo[r]) + 16*uint64(hi[r])
		rip := (s.ipOf(uint64(f.ones[r]), w) - float64(f.xbc[r])) * s.inv
		scores[r], bounds[r] = s.estimate(r, true, rip, true)
	}
}

// scoreFastGlobal is scoreFast for a global query: estimate(r, true, rip,
// false) inlined, rip as scoreAllGlobal computes it, the same operations
// in the same order (TestGlobalScoreAllMatchesRowByRow).
func (s *bitScorer) scoreFastGlobal(f *fastLayout, lo, hi []uint16, scores []float32) {
	qn, low, delta, scale, inv := s.qNorm, s.low, s.delta, s.scale, s.inv
	q2 := qn * qn
	tail := s.delta*s.sumQ + float64(s.dims)*s.low
	norms, aligns, ones, xbc := f.norms[:len(scores)], f.aligns[:len(scores)], f.ones[:len(scores)], f.xbc[:len(scores)]
	lo, hi = lo[:len(scores)], hi[:len(scores)]
	for r := range scores {
		align := float64(aligns[r])
		if align == zeroRowAlign {
			scores[r] = 0
			continue
		}
		norm := float64(norms[r])
		dist2 := norm*norm + q2
		if norm > 0 && qn > 0 && align > 0 {
			w := float64(uint64(lo[r]) + 16*uint64(hi[r]))
			dot := 2*(delta*w+low*float64(ones[r])) - tail
			rip := (dot*scale - float64(xbc[r])) * inv
			ip := rip / align
			dist2 -= 2 * norm * qn * ip
		}
		gap := norm - qn
		dist2 = max(dist2, gap*gap)
		if s.l2 {
			scores[r] = float32(-dist2)
		} else {
			scores[r] = float32(min(1, max(-1, 1-dist2/2)))
		}
	}
}

// scoreFast is estimate(r, true, ipOf(ones, sums[r]), false) for every
// row, inlined: the same operations in the same order, so the same bits
// (TestScoreAllMatchesRowByRow), without a call and two decodes per row.
func (s *bitScorer) scoreFast(f *fastLayout, sums []uint16, scores []float32) {
	qn, lo, delta, scale := s.qNorm, s.low, s.delta, s.scale
	q2 := qn * qn
	tail := s.delta*s.sumQ + float64(s.dims)*s.low
	norms, aligns, ones := f.norms[:len(scores)], f.aligns[:len(scores)], f.ones[:len(scores)]
	sums = sums[:len(scores)]
	for r := range scores {
		align := float64(aligns[r])
		if align == zeroRowAlign {
			scores[r] = 0
			continue
		}
		norm := float64(norms[r])
		dist2 := norm*norm + q2
		if norm > 0 && qn > 0 && align > 0 {
			dot := 2*(delta*float64(sums[r])+lo*float64(ones[r])) - tail
			ip := dot * scale / align
			dist2 -= 2 * norm * qn * ip
		}
		gap := norm - qn
		dist2 = max(dist2, gap*gap)
		if s.l2 {
			scores[r] = float32(-dist2)
		} else {
			scores[r] = float32(min(1, max(-1, 1-dist2/2)))
		}
	}
}

// scoreOf maps a squared distance to the metric's higher-is-better score.
// Unit rows and a unit query put cosine at 1 - d^2/2.
func scoreOf(l2 bool, dist2 float64) float64 {
	if l2 {
		return -dist2
	}
	return min(1, max(-1, 1-dist2/2))
}

// nextUp is math.Nextafter32(x, +Inf), inlined. The row loop called
// math.Max (an assembly function, never inlined) three times and
// Nextafter32 once per row, and those calls were ~28% of the 1-bit scan
// (BenchmarkBitScore profile); the builtin max and min inline and
// agree with math.Max and math.Min on every argument the loop can hold (they
// differ only on max(+Inf, NaN), and no bound term is +Inf beside a NaN).
func nextUp(x float32) float32 {
	switch {
	case x != x || x == float32(math.Inf(1)):
		return x
	case x == 0:
		return math.Float32frombits(1)
	case x > 0:
		return math.Float32frombits(math.Float32bits(x) + 1)
	}
	return math.Float32frombits(math.Float32bits(x) - 1)
}

// retained is what a decoded 1-bit payload holds in memory.
func (b *Code) retained() int {
	n := 0
	if f := b.fast; f != nil {
		n = len(f.nibs) + 2*len(f.ones) + 4*len(f.xbc) + 96 // norms and aligns: b.Norms, b.Aligns; rc is the caller's
	}
	if b.borrowed != nil {
		return n // the probe entry already charges the raw column
	}
	centroid := len(b.Centroid)
	if f := b.fast; f != nil && !f.ownCentroid {
		centroid = 0
	}
	return n + 8*len(b.Words) + 4*(centroid+len(b.Norms)+len(b.Aligns)) + 32
}
