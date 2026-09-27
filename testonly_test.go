package ivfq

import (
	"math"
	"sort"

	"github.com/bits-and-blooms/bitset"
	"github.com/go-simd/floats"
)

// Test-only helpers: nothing outside this package's tests calls them.

// wrapRow is the in-memory view of one packed []uint64 code row: bitset.From
// aliases the words, it does not copy them. The on-disk layout stays []uint64.
func wrapRow(words []uint64) *bitset.BitSet {
	return bitset.From(words)
}

func kernelDot(a, b []float32) float32 { return floats.Float32Dot(a, b) }

func kernelL2(a, b []float32) float32 {
	d := floats.Float32Distance(a, b)
	return d * d
}

func kernelCosine(a, b []float32) float32 {
	if s := floats.Float32CosineSimilarity(a, b); s == s {
		return s
	}
	return 0
}

func kernelNorm(a []float32) float32 {
	return float32(math.Sqrt(float64(floats.Float32Dot(a, a))))
}

// Rerank scores candidate IDs exactly and returns the best k, with lower IDs
// winning equal scores. exactDistance must return a higher-is-better score.
func Rerank(candidates []int, exactDistance func(int) float32, k int) []int {
	r := make([]ranked, len(candidates))
	for i, id := range candidates {
		r[i] = ranked{id, exactDistance(id)}
	}
	sort.Slice(r, func(i, j int) bool { return r[i].score > r[j].score || r[i].score == r[j].score && r[i].id < r[j].id })
	k = min(max(k, 0), len(r))
	out := make([]int, k)
	for i := range out {
		out[i] = r[i].id
	}
	return out
}

// Distance is L2, Cosine, or InnerProduct over a and b.
func Distance[T Number](m Metric, a, b Vector[T]) T {
	switch m {
	case L2:
		return a.L2(b)
	case Cosine:
		return a.Cosine(b)
	default:
		return a.Dot(b)
	}
}

func (v Vector[T]) Dot(w Vector[T]) T {
	n := min(len(v), len(w))
	if n == 0 {
		return 0
	}
	switch any(v).(type) {
	case Vector[float32]:
		return T(kernelDot(asF32(v[:n]), asF32(w[:n])))
	case Vector[float64]:
		var s float64
		a, b := asF64(v[:n]), asF64(w[:n])
		for i := 0; i < n; i++ {
			s += a[i] * b[i]
		}
		return T(s)
	default:
		var s int32
		for i := 0; i < n; i++ {
			s += asI32(v[i]) * asI32(w[i])
		}
		return T(s)
	}
}

// L2 is squared Euclidean distance.
func (v Vector[T]) L2(w Vector[T]) T {
	n := min(len(v), len(w))
	if n == 0 {
		return 0
	}
	switch any(v).(type) {
	case Vector[float32]:
		return T(kernelL2(asF32(v[:n]), asF32(w[:n])))
	case Vector[float64]:
		var s float64
		a, b := asF64(v[:n]), asF64(w[:n])
		for i := 0; i < n; i++ {
			d := a[i] - b[i]
			s += d * d
		}
		return T(s)
	default:
		var s int32
		for i := 0; i < n; i++ {
			d := asI32(v[i]) - asI32(w[i])
			s += d * d
		}
		return T(s)
	}
}

func (v Vector[T]) Cosine(w Vector[T]) T {
	n := min(len(v), len(w))
	if n == 0 {
		return 0
	}
	switch any(v).(type) {
	case Vector[float32]:
		return T(kernelCosine(asF32(v[:n]), asF32(w[:n])))
	default:
		na, nb := v[:n].Norm(), w[:n].Norm()
		if na == 0 || nb == 0 {
			return 0
		}
		return v[:n].Dot(w[:n]) / (na * nb)
	}
}

func (v Vector[T]) Norm() T {
	switch any(v).(type) {
	case Vector[float32]:
		return T(kernelNorm([]float32(asF32(v))))
	case Vector[float64]:
		return T(math.Sqrt(float64(v.Dot(v))))
	default:
		return T(math.Sqrt(float64(asI32(v.Dot(v)))))
	}
}

func (v Vector[T]) Normalized() Vector[T] {
	out := make(Vector[T], len(v))
	n := v.Norm()
	if n == 0 {
		copy(out, v)
		return out
	}
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

// MaxSim is ColBERT-style late interaction: each query token's best score
// against the document tokens, summed.
func (q Rows[T]) MaxSim(m Metric, doc Rows[T]) T {
	var sum T
	for _, token := range q {
		var best T
		for i, d := range doc {
			s := Distance(m, token, d)
			if i == 0 || s > best {
				best = s
			}
		}
		sum += best
	}
	return sum
}

func (s Sparse[T]) Dot(t Sparse[T]) T {
	var sum T
	i, j := 0, 0
	for i < len(s.Index) && j < len(t.Index) {
		switch {
		case s.Index[i] < t.Index[j]:
			i++
		case s.Index[i] > t.Index[j]:
			j++
		default:
			sum += Vector[T]{s.Value[i]}.Dot(Vector[T]{t.Value[j]})
			i++
			j++
		}
	}
	return sum
}

func asF32[T Number](v Vector[T]) []float32 { return []float32(any(v).(Vector[float32])) }

func asF64[T Number](v Vector[T]) []float64 { return []float64(any(v).(Vector[float64])) }

func asI32[T Number](x T) int32 {
	switch y := any(x).(type) {
	case int8:
		return int32(y)
	case int16:
		return int32(y)
	case int32:
		return y
	case uint8:
		return int32(y)
	case uint16:
		return int32(y)
	case float32:
		return int32(y)
	case float64:
		return int32(y)
	default:
		return 0
	}
}
