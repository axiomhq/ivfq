package recall

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"
)

// source is a Source whose approximate arm returns a set share of the true
// ids.
type source struct {
	total, cached int
	recall        float64
	reranked      int64
	exact         bool
	err           error
	at            *Knobs // the knobs Search was called with
	calls         []string
}

func (s *source) Clusters() (int, int) {
	s.calls = append(s.calls, "clusters")
	return s.total, s.cached
}

func (s *source) Search(_ context.Context, at Knobs) ([]string, bool, error) {
	s.calls = append(s.calls, "search")
	if s.at != nil {
		*s.at = at
	}
	truth := ids(100)
	return truth[:int(s.recall*100+0.5)], s.exact, s.err
}

func (s *source) Exact(context.Context) ([]string, error) {
	s.calls = append(s.calls, "exact")
	return ids(100), nil
}

func (s *source) Reranked() int64 {
	s.calls = append(s.calls, "reranked")
	return s.reranked
}

func ids(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = strconv.Itoa(i)
	}
	return out
}

var (
	plain    = Key{Index: "x", Field: "vector"}
	filtered = Key{Index: "x", Field: "vector", Filtered: true}
)

// observe lands one sample of recall r on k over 400 clusters (default
// probes 98, ceiling 196).
func observe(t *testing.T, c *Controller, k Key, r float64, reranked int64) {
	t.Helper()
	ok, err := c.Measure(context.Background(), k, &source{total: 400, cached: 400, recall: r, reranked: reranked})
	if err != nil || !ok {
		t.Fatalf("sample did not land: %v %v", ok, err)
	}
}

func state(t *testing.T, c *Controller, k Key) State {
	t.Helper()
	for _, s := range c.States() {
		if s.Key == k {
			return s
		}
	}
	t.Fatalf("no state for %+v", k)
	return State{}
}

// TestControllerRules pins the arithmetic: the sample gate, the ladders,
// and the hard floor that reverts a move which cost more recall than the
// band allows.
func TestControllerRules(t *testing.T) {
	c := New(Config{Target: 0.95, MoveSamples: 3})
	// Inside the band: no move, however many samples land.
	for range 10 {
		observe(t, c, plain, 0.96, 0)
	}
	if got := state(t, c, plain); got.Moves != 0 || got.Probes != 98 || got.Depth != Unbounded {
		t.Fatalf("inside the band the controller moved: %+v", got)
	}
	// Above the band: probes halve first, and only when they are at the
	// floor does the rerank depth come down. One knob per move, in order.
	step := func() State {
		t.Helper()
		before := state(t, c, plain)
		for range 10 {
			observe(t, c, plain, 1.0, 1)
			if got := state(t, c, plain); got.Moves != before.Moves {
				return got
			}
		}
		t.Fatalf("ten samples above the band moved nothing from %+v", before)
		return before
	}
	for _, probes := range []int{49, 24, 16} {
		if got := step(); got.Probes != probes || got.Depth != Unbounded {
			t.Fatalf("probes %d and depth %v, want %d probes at an unbounded depth", got.Probes, got.Depth, probes)
		}
	}
	// A depth is only a knob when the rerank has read something since the
	// last move: with nothing read, samples above the band move nothing.
	before := state(t, c, plain)
	for range 10 {
		observe(t, c, plain, 1.0, 0)
	}
	if got := state(t, c, plain); got.Moves != before.Moves || got.Depth != Unbounded {
		t.Fatalf("a depth the rerank never used was spent: %+v", got)
	}
	// At the 16-probe floor the 64-, 32- and 16-cluster rungs are no-ops,
	// so the ladder steps past them to 8 and then to none.
	for _, depth := range []Depth{8, None} {
		if got := step(); got.Depth != depth || got.Probes != MinProbes {
			t.Fatalf("depth %v at %d probes, want %v at the %d floor", got.Depth, got.Probes, depth, MinProbes)
		}
	}
	// Two samples of the new setting below target - 0.02 revert the last
	// lowering move and pin the value it came from. One is not enough.
	observe(t, c, plain, 0.0, 0)
	if got := state(t, c, plain); got.Depth != None {
		t.Fatalf("one sample pinned a floor: %v", got.Depth)
	}
	observe(t, c, plain, 0.0, 0)
	if got := state(t, c, plain); got.Depth != 8 {
		t.Fatalf("a floor breach must undo the depth step, got %v", got.Depth)
	}
	for range 20 {
		observe(t, c, plain, 1.0, 1)
	}
	if got := state(t, c, plain); got.Depth != 8 {
		t.Fatalf("the pinned depth floor was crossed again: %v", got.Depth)
	}
	if got := c.Tuned(plain); got != (Knobs{Probes: MinProbes, Depth: 8}) {
		t.Fatalf("Tuned %+v disagrees with the published state", got)
	}
}

// TestFilteredIsolationAndBounds: filtered samples tune probes only, between
// MinProbes and twice the default, and the two classes never move each
// other.
func TestFilteredIsolationAndBounds(t *testing.T) {
	c := New(Config{Target: .95, MoveSamples: 3})
	for range 30 {
		observe(t, c, filtered, 1, 0)
	}
	if got := state(t, c, filtered); got.Probes != MinProbes || got.Depth != Unbounded {
		t.Fatalf("filtered floor/depth: %+v", got)
	}
	if got := c.Tuned(plain); got != (Knobs{}) {
		t.Fatalf("filtered samples changed the unfiltered knobs: %+v", got)
	}
	if _, ok := c.State("x"); ok {
		t.Fatal("filtered samples published an unfiltered state")
	}
	for range 60 {
		observe(t, c, filtered, .1, 0)
	}
	low := state(t, c, filtered)
	if low.Probes != 196 || low.Depth != Unbounded {
		t.Fatalf("filtered ceiling/depth: %+v", low)
	}
	for range 30 {
		observe(t, c, plain, 1, 1)
	}
	if got := state(t, c, filtered); got != low {
		t.Fatalf("unfiltered samples changed filtered state: %+v -> %+v", low, got)
	}
	if got := c.Tuned(Key{Index: "x", Field: "other", Filtered: true}); got.Probes != 0 {
		t.Fatalf("filtered probes leaked to another field: %+v", got)
	}
}

// TestNeverBelowTheFloor: however far above the band recall sits, probes
// stop at MinProbes and depth at none.
func TestNeverBelowTheFloor(t *testing.T) {
	c := New(Config{Target: .9, MoveSamples: 1})
	for range 500 {
		observe(t, c, plain, 1, 1)
	}
	if got := c.Tuned(plain); got.Probes != MinProbes || got.Depth != None {
		t.Fatalf("knobs %+v, want %d probes and no rerank", got, MinProbes)
	}
}

// TestMeasureOnlyOnWarmData: under 99% of clusters cached, neither arm runs
// and nothing is observed.
func TestMeasureOnlyOnWarmData(t *testing.T) {
	c := New(Config{})
	for _, tc := range []struct{ total, cached int }{{0, 0}, {1000, 989}} {
		src := &source{total: tc.total, cached: tc.cached, recall: 1}
		if ok, err := c.Measure(context.Background(), plain, src); ok || !errors.Is(err, ErrCold) {
			t.Fatalf("%d of %d cached: %v %v, want ErrCold", tc.cached, tc.total, ok, err)
		}
		if len(src.calls) != 1 {
			t.Fatalf("a cold index ran %v", src.calls)
		}
	}
	src := &source{total: 1000, cached: 990, recall: 1}
	if ok, err := c.Measure(context.Background(), plain, src); !ok || err != nil {
		t.Fatalf("99%% cached: %v %v", ok, err)
	}
	if want := []string{"clusters", "search", "exact", "reranked"}; !slices.Equal(src.calls, want) {
		t.Fatalf("calls %v, want %v", src.calls, want)
	}
}

// TestMeasureSkips: a failed arm or an approximate arm that answered exactly
// lands nothing.
func TestMeasureSkips(t *testing.T) {
	c := New(Config{})
	boom := errors.New("boom")
	if ok, err := c.Measure(context.Background(), plain, &source{total: 1, cached: 1, err: boom}); ok || !errors.Is(err, boom) {
		t.Fatalf("failed arm: %v %v", ok, err)
	}
	src := &source{total: 1, cached: 1, exact: true}
	if ok, err := c.Measure(context.Background(), filtered, src); ok || err != nil {
		t.Fatalf("exact arm: %v %v", ok, err)
	}
	if len(src.calls) != 2 {
		t.Fatalf("an exact approximate arm still ran %v", src.calls)
	}
	if got := c.States(); len(got) != 0 {
		t.Fatalf("skipped samples published %+v", got)
	}
}

// TestMeasureRunsAtTheTunedKnobs: the approximate arm runs at what Tuned
// returns, and a filtered arm always at an unbounded depth.
func TestMeasureRunsAtTheTunedKnobs(t *testing.T) {
	c := New(Config{Depth: 16})
	var at Knobs
	c.Set(plain, Knobs{Probes: 20, Depth: 8})
	c.Set(filtered, Knobs{Probes: 30, Depth: None})
	if _, err := c.Measure(context.Background(), plain, &source{total: 1, cached: 1, at: &at}); err != nil || at != (Knobs{20, 8}) {
		t.Fatalf("unfiltered arm ran at %+v (%v)", at, err)
	}
	if _, err := c.Measure(context.Background(), filtered, &source{total: 1, cached: 1, at: &at}); err != nil || at != (Knobs{30, Unbounded}) {
		t.Fatalf("filtered arm ran at %+v (%v)", at, err)
	}
	if got := c.Tuned(Key{Index: "y", Field: "vector"}); got != (Knobs{Depth: 16}) {
		t.Fatalf("an untuned index runs at %+v, want the configured depth", got)
	}
}

// TestOfferIsOnePercent: 1,000 queries offer 10; an idle index none; tuning
// off offers nothing.
func TestOfferIsOnePercent(t *testing.T) {
	c := New(Config{})
	n := 0
	for range 1000 {
		if c.Offer("x") {
			n++
		}
	}
	if n != 10 {
		t.Fatalf("1,000 queries offered %d, want 10", n)
	}
	off := New(Config{Target: 1, SampleRate: 1})
	if off.Offer("x") || off.On() {
		t.Fatal("tuning off offered a sample")
	}
	if got := off.Tuned(plain); got != (Knobs{}) {
		t.Fatalf("tuning off returned %+v", got)
	}
}

// TestDueIsPerIndexOnTheClock: one measurement per index per MinInterval,
// on the Config's clock; a negative interval never limits.
func TestDueIsPerIndexOnTheClock(t *testing.T) {
	now := time.Unix(0, 0)
	c := New(Config{Now: func() time.Time { return now }})
	if !c.Due("x") || c.Due("x") || !c.Due("y") {
		t.Fatal("first attempt per index must pass and the second fail")
	}
	now = now.Add(DefaultMinInterval - time.Nanosecond)
	if c.Due("x") {
		t.Fatal("due before the interval")
	}
	now = now.Add(time.Nanosecond)
	if !c.Due("x") {
		t.Fatal("not due after the interval")
	}
	c.Forget("y")
	if !c.Due("y") {
		t.Fatal("a forgotten index kept its rate limit")
	}
	free := New(Config{MinInterval: -1})
	for range 2 {
		if !free.Due("x") {
			t.Fatal("a negative interval limited")
		}
	}
}

// TestFieldChangeStartsOver: an index tunes one field; measuring another
// resets it, and Forget drops everything.
func TestFieldChangeStartsOver(t *testing.T) {
	c := New(Config{})
	c.Set(plain, Knobs{Probes: 20})
	observe(t, c, Key{Index: "x", Field: "other"}, 0.96, 0)
	if got := c.Tuned(plain); got != (Knobs{}) {
		t.Fatalf("the old field kept %+v", got)
	}
	if s, ok := c.State("x"); !ok || s.Field != "other" || s.Last != 0.96 {
		t.Fatalf("state %+v %v", s, ok)
	}
	c.Forget("x")
	if _, ok := c.State("x"); ok || len(c.States()) != 0 {
		t.Fatal("Forget kept state")
	}
}

// TestConcurrentUse is for -race: every method from several goroutines.
func TestConcurrentUse(t *testing.T) {
	c := New(Config{MinInterval: -1, SampleRate: 1, MoveSamples: 1})
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			k := Key{Index: strconv.Itoa(g % 2), Field: "vector", Filtered: g%3 == 0}
			for i := range 200 {
				c.Offer(k.Index)
				c.Due(k.Index)
				c.Tuned(k)
				if _, err := c.Measure(context.Background(), k, &source{total: 400, cached: 400, recall: float64(i%3) / 2, reranked: 1}); err != nil {
					t.Error(err)
				}
				c.States()
				c.State(k.Index)
				if i%50 == 0 {
					c.Forget(k.Index)
				}
			}
		})
	}
	wg.Wait()
	if c.Moves() == 0 {
		t.Fatal("no moves under load")
	}
}

func TestOverlap(t *testing.T) {
	for _, tc := range []struct {
		ann, exact []string
		want       float64
	}{
		{nil, nil, 1},
		{nil, []string{"a"}, 0},
		{[]string{"b", "a"}, []string{"a", "c"}, 0.5},
	} {
		if got := Overlap(tc.ann, tc.exact); got != tc.want {
			t.Errorf("Overlap(%v, %v) = %v, want %v", tc.ann, tc.exact, got, tc.want)
		}
	}
}

func TestDepth(t *testing.T) {
	for _, tc := range []struct {
		d            Depth
		probed, want int
		s            string
	}{
		{Unbounded, 40, 40, "unbounded"},
		{None, 40, 0, "none"},
		{8, 40, 8, "8 hoods"},
		{64, 40, 40, "64 hoods"},
	} {
		if got := tc.d.Clusters(tc.probed); got != tc.want || tc.d.String() != tc.s {
			t.Errorf("%v: Clusters(%d) = %d, String %q", tc.d, tc.probed, got, tc.d.String())
		}
	}
}
