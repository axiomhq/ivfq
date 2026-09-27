package ivfq

import (
	"testing"
)

func TestDefaultPolicyPinned(t *testing.T) {
	p := DefaultPolicy{}
	if _, ok := p.Bootstrap(63); ok {
		t.Fatal("bootstrap below min")
	}
	if k, ok := p.Bootstrap(64); !ok || k != 8 {
		t.Fatalf("k=%d ok=%v", k, ok)
	}
}

// TestNoPolicyTriggerSeesTheWholeIndex is the interface-level statement of
// what this package gave up on 2026-09-05: every trigger past Bootstrap takes
// []ClusterStat and nothing else. There is no argument through which a policy
// could learn N, and therefore no way for one to ask for a pass over N. The
// drift trigger that used to (ReclusterK, taking an IndexStats of additive
// and live totals) is gone; an explicit recluster is the only verb that still
// rebuilds an index on request.
//
// This is a compile-time property, so the test is a compile-time assertion:
// the Policy interface is satisfied by a type whose only per-index input is a
// cluster slice. If a whole-index trigger is ever added back, this stops
// building and whoever added it has to read the paragraph above.
func TestNoPolicyTriggerSeesTheWholeIndex(t *testing.T) {
	var _ Policy = clusterOnlyPolicy{}
}

type clusterOnlyPolicy struct{ DefaultPolicy }

// pad makes n filler hoods of the given size, so a table case can state a
// realistic total (and therefore a realistic per-hood target) without
// spelling out sixteen clusters.
func pad(n, count int) []ClusterStat {
	out := make([]ClusterStat, n)
	for i := range out {
		out[i] = ClusterStat{ID: 100 + i, Count: count, Radius: .1}
	}
	return out
}

// padRadius sets every filler hood's radius to r.
func padRadius(in []ClusterStat, r float64) []ClusterStat {
	for i := range in {
		in[i].Radius = r
	}
	return in
}

// TestSplitTargetRatioBoundary pins SplitTarget's ratio-gate boundary (T1
// review minor c, optional): count exactly equal to splitFactor×mean must
// NOT split (strict >); one more must.
func TestSplitTarget(t *testing.T) {
	p := DefaultPolicy{}
	tests := []struct {
		name string
		in   []ClusterStat
		id   int
		ok   bool
	}{
		// Both radius cases are stated against a padded index so the SIZE
		// trigger stays quiet and they still test what they were written to
		// test. Unpadded, a two-hood {80, 8} index has a target of 9 and 80
		// is a legitimate split under fixed-size hoods.
		{"tight fat", append([]ClusterStat{{ID: 1, Count: 80}, {ID: 2, Count: 8}}, pad(14, 64)...), 0, false},
		{"below minimum", append([]ClusterStat{{ID: 1, Count: 7, Radius: 100}, {ID: 2, Count: 8, Radius: .1}}, pad(3, 8)...), 0, false},
		{"sick", []ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
		// Padded at radius .2 so the mean stays .2 and the size trigger quiet.
		{"strict boundary", append([]ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: 0}}, padRadius(pad(14, 8), .2)...), 0, false},
		{"cold sick ignored", append([]ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1, Hits: 1}, {ID: 3, Count: 8, Radius: .1}}, pad(14, 8)...), 0, false},
		{"hot sick", []ClusterStat{{ID: 1, Count: 8, Radius: .4, Hits: 1}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
		{"unseen fallback", []ClusterStat{{ID: 1, Count: 8, Radius: .4}, {ID: 2, Count: 8, Radius: .1}, {ID: 3, Count: 8, Radius: .1}, {ID: 4, Count: 8, Radius: .1}, {ID: 5, Count: 8, Radius: .1}}, 1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id, ok := p.SplitTarget(tt.in)
			if id != tt.id || ok != tt.ok {
				t.Fatalf("got (%d,%v), want (%d,%v)", id, ok, tt.id, tt.ok)
			}
		})
	}
}

// TestSplitClusterRejectsUnknownTarget: a policy returning a split id that
// names no real cluster must surface as a CompactNow error (the caller
// already log-and-retries — see startCompactor), never a panic.
