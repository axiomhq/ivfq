package ivfq

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// fillRows must produce fillRow's codes bit for bit: same bits, and the
// same float32 norms and alignments, for l2 and cosine, zero rows, rows
// equal to the centroid and blocks cut short.
func TestFillRowsMatchesFillRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 9))
	for _, dims := range []int{1, 2, 3, 16, 100, 128, 129} {
		for _, metric := range []string{"l2", "cosine"} {
			for _, rows := range []int{0, 1, fillBlock - 1, fillBlock, 3*fillBlock + 5} {
				vectors := make([][]float32, rows)
				for i := range vectors {
					v := make([]float32, dims)
					switch i % 7 {
					case 3: // zero vector
					case 5: // an i8-like row
						for j := range v {
							v[j] = float32(rng.IntN(256) - 128)
						}
					default:
						for j := range v {
							v[j] = float32(rng.NormFloat64() * 10)
						}
					}
					vectors[i] = v
				}
				if rows > 2 {
					vectors[1] = slices.Clone(vectors[2]) // duplicates pull a row onto the mean when rows == 2
				}
				got, err := quantizeBits(vectors, dims, metric, 11)
				if err != nil {
					t.Fatal(err)
				}
				want := *got.Code
				want.Words = make([]uint64, len(want.Words))
				want.Norms = make([]float32, len(want.Norms))
				want.Aligns = slices.Clone(got.Code.Aligns)
				rot := rotationFor(11, dims)
				scratch := make([]float32, dims)
				for i, v := range vectors {
					row := workRow(v, metric == "cosine")
					if row == nil {
						continue
					}
					want.Aligns[i] = 0
					want.fillRow(i, row, rot, scratch)
				}
				if !slices.Equal(got.Code.Words, want.Words) || !bitsEqual(got.Code.Norms, want.Norms) || !bitsEqual(got.Code.Aligns, want.Aligns) {
					t.Fatalf("dims=%d %s rows=%d: fillRows differs from fillRow", dims, metric, rows)
				}
			}
		}
	}
	// A hood of one row: the row is its own centroid, zero norm, no bits.
	q, err := quantizeBits([][]float32{{1, 2, 3}}, 3, "l2", 5)
	if err != nil || q.Code.Norms[0] != 0 || q.Code.Aligns[0] != 0 || q.Code.Words[0] != 0 {
		t.Fatalf("single row: %+v %v", q.Code, err)
	}
}

func bitsEqual(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// Both block kernels are Apply on each row of the block, bit for bit.
func TestRotatePairsMatchesApply(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, dims := range []int{2, 17, 128} {
		rot := newRotation(99, dims)
		rows := make([][]float32, fillBlock)
		block := make([]float32, dims*fillBlock)
		for r := range rows {
			rows[r] = make([]float32, dims)
			for j := range rows[r] {
				rows[r][j] = float32(rng.NormFloat64())
				block[j*fillBlock+r] = rows[r][j]
			}
		}
		generic := slices.Clone(block)
		rotatePairsGeneric(rot.pairs, generic)
		rot.applyBlock(block)
		for r, row := range rows {
			rot.Apply(row)
			for j, x := range row {
				if math.Float32bits(block[j*fillBlock+r]) != math.Float32bits(x) || math.Float32bits(generic[j*fillBlock+r]) != math.Float32bits(x) {
					t.Fatalf("dims=%d row %d dim %d: block %v generic %v, Apply %v", dims, r, j, block[j*fillBlock+r], generic[j*fillBlock+r], x)
				}
			}
		}
	}
}
