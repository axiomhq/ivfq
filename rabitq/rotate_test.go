package rabitq

import (
	"github.com/axiomhq/ivfq/internal/simd"
	"math"
	"math/rand/v2"
	"testing"
)

func TestSeedNeverZero(t *testing.T) {
	if Seed() == 0 {
		t.Fatal("empty seed")
	}
	if Seed("ns", "vector") == 0 {
		t.Fatal("named seed")
	}
}

// The rotation is orthogonal at any width, exactly enough: norms and dot
// products survive, for the widths whose blocks are all sizes (100 = 64+32+4).
func TestRotationOrthogonal(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, dims := range []int{2, 3, 7, 33, 100, 128, 768, 1000, 1536} {
		rot := NewRotation(Seed("t", "o"), dims)
		a, b := make([]float32, dims), make([]float32, dims)
		var na, ab float64
		for i := range a {
			a[i], b[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			na += float64(a[i]) * float64(a[i])
			ab += float64(a[i]) * float64(b[i])
		}
		rot.Apply(a)
		rot.Apply(b)
		var na2, ab2 float64
		for i := range a {
			na2 += float64(a[i]) * float64(a[i])
			ab2 += float64(a[i]) * float64(b[i])
		}
		if math.Abs(na-na2) > 1e-4*na || math.Abs(ab-ab2) > 1e-4*math.Sqrt(na*na) {
			t.Fatalf("dims=%d: |a|^2 %v -> %v, a.b %v -> %v", dims, na, na2, ab, ab2)
		}
	}
}

// A one-hot vector, the worst case for a corpus aligned to the axes, must
// spread: no rotated coordinate may hold much of its energy.
func TestRotationSpreadsOneHot(t *testing.T) {
	for _, dims := range []int{33, 100, 128, 768, 1000} {
		rot := NewRotation(Seed("t", "s"), dims)
		var worst float64
		for hot := 0; hot < dims; hot += 7 {
			v := make([]float32, dims)
			v[hot] = 1
			rot.Apply(v)
			for _, x := range v {
				worst = max(worst, float64(x)*float64(x))
			}
		}
		if worst > 40/float64(dims) { // uniform would be ~ 1/dims, tail ~ 2 ln(dims)/dims
			t.Fatalf("dims=%d: one coordinate holds %.3f of a one-hot's energy", dims, worst)
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5}
	var n0 float64
	for _, x := range v {
		n0 += float64(x) * float64(x)
	}
	NewRotation(Seed("t", "r"), len(v)).Apply(v)
	var n1 float64
	for _, x := range v {
		n1 += float64(x) * float64(x)
	}
	if math.Abs(n0-n1) > 1e-5 {
		t.Fatalf("norm %v -> %v", n0, n1)
	}
}

func TestApplyBlockMatchesApply(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, dims := range []int{2, 3, 17, 100, 128, 1000, 1536} {
		rot := NewRotation(99, dims)
		rows := make([][]float32, simd.FillBlock)
		block := make([]float32, dims*simd.FillBlock)
		for r := range rows {
			rows[r] = make([]float32, dims)
			for j := range rows[r] {
				rows[r][j] = float32(rng.NormFloat64())
				block[j*simd.FillBlock+r] = rows[r][j]
			}
		}
		rot.applyBlock(block)
		for r, row := range rows {
			rot.Apply(row)
			for j, x := range row {
				if math.Float32bits(block[j*simd.FillBlock+r]) != math.Float32bits(x) {
					t.Fatalf("dims=%d row %d dim %d: block %v, Apply %v", dims, r, j, block[j*simd.FillBlock+r], x)
				}
			}
		}
	}
}
