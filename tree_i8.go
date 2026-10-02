package ivfq

import (
	"math"

	"github.com/axiomhq/ivfq/internal/simd"
)

// nodeI8 is a holder's leaves as int8 residuals from the holder's mean:
// leaves of one holder are close, so their residuals are short and so is
// their rounding error. Per leaf, one symmetric scale, the rounding
// error's norm ||r - r~||, the norm of what the codes stand for ||r~||, and
// <mean, r - r~>, the part of the rounding error a query's dot product
// with the mean already knows.
type nodeI8 struct {
	mean                      []float32
	mean2                     float64 // ||mean||^2
	codes                     []int8
	scale, err, norm8, offset []float32
}

// QuantizeI8 gives every leaf holder an int8 copy of its leaves for
// EvaluationsBeamI8: a quarter of the bytes the leaf scan reads, which at
// 768 dimensions was 12.7 MB a query. It is derived, never serialized, and
// dropped by any change to the tree (Upsert, Truncate), after which
// EvaluationsBeamI8 reads the float32 leaves until QuantizeI8 runs again.
// It writes t, so call it before t is shared.
func (t *Tree) QuantizeI8() {
	for i := range t.nodes {
		n := &t.nodes[i]
		if len(n.children) > 0 || len(n.leaves) == 0 || len(n.block) != len(n.leaves)*t.dims {
			continue
		}
		rows := len(n.leaves)
		q := &nodeI8{mean: make([]float32, t.dims), codes: make([]int8, len(n.block)), scale: make([]float32, rows),
			err: make([]float32, rows), norm8: make([]float32, rows), offset: make([]float32, rows)}
		for j := range rows {
			for d, x := range n.block[j*t.dims : (j+1)*t.dims] {
				q.mean[d] += x
			}
		}
		for d := range q.mean {
			q.mean[d] /= float32(rows)
			q.mean2 += float64(q.mean[d]) * float64(q.mean[d])
		}
		r := make([]float32, t.dims)
		for j := range rows {
			for d, x := range n.block[j*t.dims : (j+1)*t.dims] {
				r[d] = x - q.mean[d]
			}
			codes := q.codes[j*t.dims : (j+1)*t.dims]
			s, e, m := quantizeI8(r, codes)
			var off float64
			for d, x := range r {
				off += float64(q.mean[d]) * (float64(x) - float64(s)*float64(codes[d]))
			}
			q.scale[j], q.err[j], q.norm8[j], q.offset[j] = s, e, m, float32(off)
		}
		n.i8 = q
	}
}

// dropI8 forgets every holder's int8 copy: the tree changed under it.
func (t *Tree) dropI8() {
	for i := range t.nodes {
		t.nodes[i].i8 = nil
	}
}

// quantizeI8 writes v rounded to int8 at a symmetric scale into dst and
// returns the scale, the rounding error's norm and the norm of what dst
// stands for. A zero vector has scale 1 and no error.
func quantizeI8(v []float32, dst []int8) (scale, err, norm8 float32) {
	var m float32
	for _, x := range v {
		m = max(m, float32(math.Abs(float64(x))))
	}
	scale = 1
	if m > 0 {
		scale = m / 127
	}
	var e2, n2 float64
	for i, x := range v {
		c := int8(max(-127, min(127, math.Round(float64(x/scale)))))
		dst[i] = c
		y := float64(scale) * float64(c)
		e2 += (float64(x) - y) * (float64(x) - y)
		n2 += y * y
	}
	return scale, float32(math.Sqrt(e2)), float32(math.Sqrt(n2))
}

// EvaluationsBeamI8 is EvaluationsBeam, the same leaves in the same order,
// with the leaf scan read through the holders' int8 copies (QuantizeI8).
// The beam over internal nodes is EvaluationsBeam's own, in float32, so it
// visits the same holders. Each scanned leaf gets an int8 distance and an
// exact bound on its error,
//
//	|<q,c> - <q~,c~>| <= ||q|| ||c - c~|| + ||q - q~|| ||c~||
//
// and its float32 distance is computed, as EvaluationsBeam computes it,
// only for the probe nearest by int8 and for any other leaf whose lower
// bound is within the probe-th float32 distance so far. The count is the
// float32 distances computed, internal nodes included. A tree without its
// int8 copies (not quantized, or changed since) is searched by
// EvaluationsBeam.
func (t Tree) EvaluationsBeamI8(q []float32, beam, probe int) ([]int, int) {
	if t.root < 0 || probe <= 0 || beam <= 0 || len(q) != t.dims {
		return nil, 0
	}
	w := workspaces.Get().(*treeWorkspace)
	defer workspaces.Put(w)
	evals := t.beamHolders(w, q, beam)
	for _, id := range w.holders {
		if t.nodes[id].i8 == nil {
			return t.nearestBeam(q, beam, probe)
		}
	}
	qq := Dot(q, q)
	if cap(w.q8) < t.dims {
		w.q8 = make([]int8, t.dims)
	}
	q8 := w.q8[:t.dims]
	qs, qe, _ := quantizeI8(q, q8)
	// Every scanned leaf: its int8 distance (next), and per candidate its
	// error bound and float32 norm, alongside.
	w.next, w.slack, w.lnorm = w.next[:0], w.slack[:0], w.lnorm[:0]
	for _, id := range w.holders {
		n := &t.nodes[id]
		if cap(w.dots8) < len(n.leaves) {
			w.dots8 = make([]int32, len(n.leaves))
		}
		dots := w.dots8[:len(n.leaves)]
		simd.DotsI8(q8, n.i8.codes, dots)
		qm := float64(Dot(q, n.i8.mean))
		qmd := math.Sqrt(max(0, float64(qq)+n.i8.mean2-2*qm)) // ||q - mean||
		for j, leaf := range n.leaves {
			// <q,c> = <q,mean> + <q~,r~> + <mean, r-r~> + <q-q~, r~> + <q-mean, r-r~>;
			// the last two are the error, bounded by Cauchy-Schwarz.
			dot := qm + float64(qs)*float64(n.i8.scale[j])*float64(dots[j]) + float64(n.i8.offset[j])
			d := float64(qq) + float64(n.norms[j]) - 2*dot
			// Twice the dot product's bound, and a margin for the float32
			// rounding of the distance EvaluationsBeam computes, which this
			// bound in real arithmetic does not cover.
			slack := 2*(float64(qe)*float64(n.i8.norm8[j])+qmd*float64(n.i8.err[j])) +
				1e-5*(float64(qq)+float64(n.norms[j]))
			w.next = append(w.next, rankedNode{leaf, float32(d)})
			w.slack = append(w.slack, slack)
			w.lnorm = append(w.lnorm, n.norms[j])
		}
	}
	count := min(probe, len(w.next))
	if count == 0 {
		return nil, evals
	}
	// T: the count-th int8 distance. Every leaf at or under it is measured
	// first, which sets the cut; then any other leaf that its bound cannot
	// rule out against the cut. The cut only falls as leaves are measured.
	w.best = selectNearest(w.best[:0], w.next, count)
	threshold := w.best[len(w.best)-1].d
	out := w.out8[:0]
	measure := func(i int) {
		evals++
		c := w.next[i]
		d := float64(qq) + float64(w.lnorm[i]) - 2*float64(Dot(q, t.leaves[c.id]))
		out = insertNearest(out, rankedNode{c.id, float32(d)}, count)
	}
	for i, c := range w.next {
		if c.d <= threshold {
			measure(i)
		}
	}
	for i, c := range w.next {
		if c.d > threshold && (len(out) < count || float64(c.d)-w.slack[i] <= float64(out[len(out)-1].d)) {
			measure(i)
		}
	}
	w.out8 = out
	ids := make([]int, len(out))
	for i, x := range out {
		ids[i] = x.id
	}
	return ids, evals
}

// beamHolders is nearestInto's beam over the internal nodes: w.holders is
// the leaf holders it reaches, and it returns the distances computed.
func (t Tree) beamHolders(w *treeWorkspace, q []float32, beam int) int {
	w.active = append(w.active[:0], t.root)
	w.holders = w.holders[:0]
	evals := 0
	qq := Dot(q, q)
	for len(w.active) > 0 {
		w.best = w.best[:0]
		for _, id := range w.active {
			n := &t.nodes[id]
			if len(n.children) == 0 {
				w.holders = append(w.holders, id)
				continue
			}
			dots := w.dotsOf(q, n.block, len(n.children))
			for j, child := range n.children {
				d := float64(qq) + float64(n.norms[j]) - 2*float64(dots[j])
				w.best = insertNearest(w.best, rankedNode{child, float32(d)}, beam)
			}
			evals += len(n.children)
		}
		if len(w.best) == 0 {
			break
		}
		w.active = w.active[:0]
		for _, x := range w.best {
			w.active = append(w.active, x.id)
		}
	}
	return evals
}
