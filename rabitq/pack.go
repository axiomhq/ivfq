package rabitq

import (
	"encoding/binary"
	"math"
	"math/bits"
)

// Packing a column for the batched scan (PackFastScan, PackFastScanFrame)
// is a block of 32 rows at a time: the block's bit words are gathered once,
// transposed into nibbles eight rows by eight bytes at a time, and, with a
// frame, summed into each row's <x_b, Rc>. The layout, the popcounts and
// every float are what a row-at-a-time pack produces, bit for bit
// (TestPackMatchesRowAtATime): the nibble transpose only moves bits, and a
// row's sum adds the same terms in the same order, rows interleaved.

// frameSums is what PackFastScanFrame accumulates per row while packing:
// rc widened to float64 once, Σ rc, the 1/sqrt(D) scale, and the output.
// rc64 holds word w's 64 values at 64w, each at its bit's de Bruijn slot
// (deBruijnSlot) rather than its bit index: a set bit's value is then one
// multiply and one load away, where bits.TrailingZeros64 is BSF below
// GOAMD64=v3, several micro-ops and slow on AMD's cores.
type frameSums struct {
	rc64       []float64
	sum, scale float64
	xbc        []float32
}

const deBruijn64 = 0x03f79d71b4ca8b09

// deBruijnSlot is where bit j's value sits in a word's 64 slots: the top
// six bits of deBruijn64 << j, distinct for each j.
func deBruijnSlot(lowBit uint64) uint64 { return lowBit * deBruijn64 >> 58 }

// newFrameSums is rc prepared for rows of words bit words.
func newFrameSums(rc []float32, words, rows int, sum, scale float64) *frameSums {
	fr := &frameSums{rc64: make([]float64, 64*words), sum: sum, scale: scale, xbc: make([]float32, rows)}
	for j, v := range rc {
		fr.rc64[64*(j/64)+int(deBruijnSlot(1<<(j%64)))] = float64(v)
	}
	return fr
}

// pack is PackFastScan, plus the frame's sums when fr is not nil.
func (b *Code) pack(fr *frameSums) {
	groups := fastGroups(b.width())
	if b.fast != nil || groups == 0 {
		return
	}
	rows, words := b.Rows(), bitWords(b.width())
	f := &fastLayout{groups: groups, blocks: (rows + 31) / 32, ones: make([]uint16, rows),
		norms: make([]float32, rows), aligns: make([]float32, rows)}
	f.nibs = make([]byte, f.blocks*groups*16)
	if b.ExBits > 0 {
		f.exNibs = make([]byte, b.ExBits*len(f.nibs))
		f.exSum = make([]uint32, rows)
	}
	x := make([]uint64, 32*words)
	for k := range f.blocks {
		n := b.loadBlock(k, words, x)
		packNibbles(x, words, groups, f.nibs[k*groups*16:(k+1)*groups*16])
		for i := range n {
			ones := 0
			for _, w := range x[i*words : (i+1)*words] {
				ones += bits.OnesCount64(w)
			}
			f.ones[32*k+i] = uint16(ones)
		}
		if fr != nil {
			fr.block(x, words, n, 32*k)
		}
		if b.ExBits > 0 {
			for i := range n {
				f.exSum[32*k+i] = uint32(f.ones[32*k+i]) << b.ExBits
			}
		}
		for p := range b.ExBits {
			clear(x)
			for i := range n {
				for g := range words {
					x[i*words+g] = b.exWord(32*k+i, p, g)
				}
			}
			plane := f.blocks * groups * 16
			packNibbles(x, words, groups, f.exNibs[p*plane+k*groups*16:p*plane+(k+1)*groups*16])
			for i := range n {
				ones := 0
				for _, w := range x[i*words : (i+1)*words] {
					ones += bits.OnesCount64(w)
				}
				f.exSum[32*k+i] += uint32(ones) << p
			}
		}
	}
	if b.borrowed != nil {
		base := bitsHeader + b.cOff
		raw := b.borrowed[base : base+8*rows]
		for r := range rows {
			f.norms[r] = float32frombits(raw[4*r:])
			f.aligns[r] = float32frombits(raw[4*rows+4*r:])
		}
	} else {
		copy(f.norms, b.Norms[:rows])
		copy(f.aligns, b.Aligns[:rows])
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
	b.Words, b.Ex, b.borrowed, b.cOff = nil, nil, nil, 0
	b.fast = f
}

// frameOf fills fr from a column already packed without a frame: the cold
// path, through the packed rows' reassembled words.
func (b *Code) frameOf(fr *frameSums) {
	rows, words := b.Rows(), bitWords(b.width())
	x := make([]uint64, 32*words)
	for k := range (rows + 31) / 32 {
		n := b.loadBlock(k, words, x)
		fr.block(x, words, n, 32*k)
	}
}

// loadBlock sets x to block k's rows of words bit words each, row-major,
// zero past the column's last row, and returns how many rows it holds.
func (b *Code) loadBlock(k, words int, x []uint64) int {
	r0 := 32 * k
	n := min(32, b.Rows()-r0)
	m := n * words
	switch {
	case b.borrowed != nil:
		off := bitsHeader + b.cOff + 8*b.rows + 8*r0*words
		raw := b.borrowed[off : off+8*m]
		for i := range x[:m] {
			x[i] = binary.LittleEndian.Uint64(raw[8*i:])
		}
	case b.Words != nil:
		copy(x[:m], b.Words[r0*words:r0*words+m])
	default:
		for i := range x[:m] {
			x[i] = b.word(r0*words + i)
		}
	}
	clear(x[m:])
	return n
}

func float32frombits(p []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(p))
}

const (
	lowNibbles  = 0x0F0F0F0F0F0F0F0F
	highNibbles = 0xF0F0F0F0F0F0F0F0
)

// packNibbles writes one block's groups (dst, groups*16 bytes) from its 32
// rows' words (x, row-major). Byte i of group g holds nibble g of row i in
// its low half and of row 16+i in its high half; nibble g is bits 4g..4g+3
// of the row, word g/16. Rows i and 16+i are first merged into one word per
// nibble parity: byte j of lo holds both rows' nibble 2j, of hi nibble
// 2j+1. Eight such words (rows 8h..8h+7) transposed as an 8x8 byte matrix
// are then eight groups' bytes 8h..8h+7, each one 64-bit store.
func packNibbles(x []uint64, words, groups int, dst []byte) {
	dst = dst[:groups*16]
	for w := range words {
		g0 := 16 * w
		if g0 >= groups {
			break
		}
		for h := range 2 {
			var lo, hi [8]uint64
			for i := range 8 {
				a, c := x[(8*h+i)*words+w], x[(16+8*h+i)*words+w]
				lo[i] = a&lowNibbles | (c&lowNibbles)<<4
				hi[i] = a>>4&lowNibbles | c&highNibbles
			}
			transpose8(&lo)
			transpose8(&hi)
			for j := range 8 {
				g := g0 + 2*j
				if g >= groups { // groups is even: g+1 < groups too
					break
				}
				binary.LittleEndian.PutUint64(dst[g*16+8*h:], lo[j])
				binary.LittleEndian.PutUint64(dst[(g+1)*16+8*h:], hi[j])
			}
		}
	}
}

// transpose8 transposes m as an 8x8 byte matrix, byte j of m[i] its entry
// (i, j): afterwards byte i of m[j] is what byte j of m[i] was.
func transpose8(m *[8]uint64) {
	for i := range 4 {
		a, c := m[i], m[i+4]
		t := (a>>32 ^ c) & 0x00000000FFFFFFFF
		m[i], m[i+4] = a^t<<32, c^t
	}
	for _, i := range [4]int{0, 1, 4, 5} {
		a, c := m[i], m[i+2]
		t := (a>>16 ^ c) & 0x0000FFFF0000FFFF
		m[i], m[i+2] = a^t<<16, c^t
	}
	for i := 0; i < 8; i += 2 {
		a, c := m[i], m[i+1]
		t := (a>>8 ^ c) & 0x00FF00FF00FF00FF
		m[i], m[i+1] = a^t<<8, c^t
	}
}

// block sets xbc for the n rows of x (row-major, words each) that start at
// row r0: <x_b, rc> = (2 Σ_{b_j=1} rc_j - Σ rc_j) / sqrt(D). Each row's
// Σ adds its set bits' rc_j in ascending j, as one float64 chain; four
// rows' chains run interleaved, so the adds overlap instead of waiting out
// one another's latency.
func (fr *frameSums) block(x []uint64, words, n, r0 int) {
	out := fr.xbc[r0 : r0+n]
	r := 0
	for ; r+4 <= n; r += 4 {
		var o0, o1, o2, o3 float64
		for w := range words {
			seg := (*[64]float64)(fr.rc64[64*w:])
			x0, x1, x2, x3 := x[r*words+w], x[(r+1)*words+w], x[(r+2)*words+w], x[(r+3)*words+w]
			for x0 != 0 && x1 != 0 && x2 != 0 && x3 != 0 {
				l0, l1, l2, l3 := x0&-x0, x1&-x1, x2&-x2, x3&-x3
				o0 += seg[deBruijnSlot(l0)]
				o1 += seg[deBruijnSlot(l1)]
				o2 += seg[deBruijnSlot(l2)]
				o3 += seg[deBruijnSlot(l3)]
				x0, x1, x2, x3 = x0^l0, x1^l1, x2^l2, x3^l3
			}
			o0 = sumBits(o0, x0, seg)
			o1 = sumBits(o1, x1, seg)
			o2 = sumBits(o2, x2, seg)
			o3 = sumBits(o3, x3, seg)
		}
		out[r] = float32((2*o0 - fr.sum) * fr.scale)
		out[r+1] = float32((2*o1 - fr.sum) * fr.scale)
		out[r+2] = float32((2*o2 - fr.sum) * fr.scale)
		out[r+3] = float32((2*o3 - fr.sum) * fr.scale)
	}
	for ; r < n; r++ {
		var on float64
		for w := range words {
			on = sumBits(on, x[r*words+w], (*[64]float64)(fr.rc64[64*w:]))
		}
		out[r] = float32((2*on - fr.sum) * fr.scale)
	}
}

// sumBits is on plus bit j's value for each set bit j of x, ascending;
// seg is the word's slots (frameSums).
func sumBits(on float64, x uint64, seg *[64]float64) float64 {
	for x != 0 {
		l := x & -x
		on += seg[deBruijnSlot(l)]
		x ^= l
	}
	return on
}
