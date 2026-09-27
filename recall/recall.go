package recall

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/axiomhq/ivfq"
)

// Depth is how many of the probed clusters the bound-pruned rerank may read
// rows from. The zero value is Unbounded: exact within the probed clusters,
// to whatever the candidate codec's bound guarantees.
type Depth int

const (
	// Unbounded lets the rerank consider every probed cluster.
	Unbounded Depth = 0
	// None skips the bound-pruned rerank: the query answers from its
	// shortlist alone.
	None Depth = -1
)

// ladder is the order the controller walks depth, shallowest first. Steps
// are multiplicative for the same reason the probe steps are: recall
// responds to the log of the rows reranked, so a linear ladder would spend a
// dozen samples crossing the range.
var ladder = []Depth{None, 8, 16, 32, 64, Unbounded}

// Clusters is how many of probed clusters the rerank may read from.
func (d Depth) Clusters(probed int) int {
	switch {
	case d == Unbounded:
		return probed
	case d < 0:
		return 0
	default:
		return min(int(d), probed)
	}
}

func (d Depth) String() string {
	switch {
	case d == Unbounded:
		return "unbounded"
	case d < 0:
		return "none"
	default:
		return strconv.Itoa(int(d)) + " hoods"
	}
}

// index reports where d sits on the ladder; an unknown value reads as
// unbounded, the safe end.
func (d Depth) index() int {
	if i := slices.Index(ladder, d); i >= 0 {
		return i
	}
	return len(ladder) - 1
}

// Knobs is what an approximate query runs at. Probes 0 means
// ivfq.DefaultNprobe of the index's cluster count.
type Knobs struct {
	Probes int
	Depth  Depth
}

// Key names one tuned state. An index tunes one vector field at a time: a
// measurement of another field starts the index over. Filtered queries tune
// their probe count on their own, and never shorten the rerank.
type Key struct {
	Index, Field string
	Filtered     bool
}

// Defaults for Config's zero fields.
const (
	// DefaultTarget sits inside a 90-100% recall band with room for the
	// controller's own band above it.
	DefaultTarget = 0.95
	// DefaultSampleRate offers 1 in 100 live queries.
	DefaultSampleRate = 100
	// DefaultMinInterval is the per-index floor between measurements,
	// whatever the query rate.
	DefaultMinInterval = 30 * time.Second
	// DefaultMoveSamples is how many samples land between two moves.
	DefaultMoveSamples = 5
	// MinProbes is the floor the controller may lower probes to. Below 16
	// recall falls faster than latency does (0.919 at 16 on SIFT1M).
	MinProbes = 16
)

const (
	// band is how far above the target recall may sit before the surplus
	// is spent. Narrower and the controller oscillates on sampling noise.
	band = 0.03
	// floorMargin is the hard floor: an average this far below the target
	// reverts the move that caused it without waiting for MoveSamples, and
	// pins the value it came from as a floor.
	floorMargin = 0.02
	// floorSamples is how many measurements of the new setting the floor
	// rule needs. One is not enough: recall@10 of one query moves in
	// tenths, and a floor is permanent for the life of the Controller.
	floorSamples = 2
	// ewma weights each sample against the running average: about 15
	// samples of memory, long enough that one query does not move a knob,
	// short enough to follow a corpus that drifts.
	ewma = 0.2
	// maxProbeFactor caps probes at this multiple of ivfq.DefaultNprobe.
	maxProbeFactor = 2
	// warmShare is how much of the index must be cached before a
	// measurement may run: the exact arm probes every cluster. A share, not
	// all of it, because skewed traffic leaves a few clusters nobody probes.
	warmShare = 0.99
)

// ErrCold is Measure refusing an index less than 99% of whose clusters are
// cached: the exact arm would be a cold scan.
var ErrCold = errors.New("recall: index is not warm")

// Config sets the Controller's policy. Zero fields take the defaults.
type Config struct {
	// Target is the recall@k the controller holds. 1 or more turns
	// tuning off: nothing is offered and Tuned returns the defaults.
	Target float64
	// SampleRate offers every SampleRate'th query of an index.
	SampleRate int
	// MinInterval is the least time between two measurements of one
	// index. Negative: no limit.
	MinInterval time.Duration
	// MoveSamples is how many samples must land between two moves.
	MoveSamples int
	// Depth is the rerank depth an untuned index starts at.
	Depth Depth
	// Now is the clock. Nil: time.Now.
	Now func() time.Time
}

// Source is one sampled query bound to one consistent view of the index.
// Measure calls Clusters, then Search, then Exact, then Reranked, each at
// most once.
type Source interface {
	// Clusters is how many clusters the exact arm would probe, and how
	// many of them are already cached.
	Clusters() (total, cached int)
	// Search runs the query approximately at the given knobs and returns
	// its top-k ids. exact reports that it answered exactly anyway (a
	// selective filter, say): such a sample measures nothing.
	Search(ctx context.Context, at Knobs) (ids []string, exact bool, err error)
	// Exact returns the true top-k ids of the same query on the same view.
	Exact(ctx context.Context) ([]string, error)
	// Reranked is how many rows live queries' bound-pruned rerank read
	// since the last call. Zero keeps the controller from shortening a
	// depth nothing used: that step could not be measured. Called only
	// for unfiltered keys.
	Reranked() int64
}

// State is one key's tuned state as a Controller publishes it.
type State struct {
	Key
	Knobs           // Probes resolved against the cluster count, never 0
	Recall  float64 // exponentially weighted recall@k
	Last    float64 // the latest sample
	Samples int
	Moves   int
	At      time.Time // when the latest sample landed
}

// Controller holds every index's tuned state. It is safe for concurrent use.
type Controller struct {
	cfg   Config
	moves atomic.Int64

	mu      sync.Mutex // guards tune, queries and attempt
	tune    map[string]*tuning
	queries map[string]uint64
	attempt map[string]time.Time
}

// tuning is one index's state; filtered is its filtered-query child.
type tuning struct {
	filtered *tuning
	field    string
	clusters int   // cluster count the probe bounds are computed from
	probes   int   // absolute; 0 = ivfq.DefaultNprobe
	depth    Depth // 0 = unbounded
	recall   float64
	last     float64
	samples  int
	moves    int
	at       time.Time

	sinceMove  int
	probeFloor int // proven floor: never lower probes below this
	depthFloor int // proven floor: never go shallower than ladder[i]
	// The last move, so a sample below the hard floor can undo exactly it.
	// Only a move that lowered a knob is reverted; raising one cannot be
	// what cost the recall.
	lastKnob   string
	lastDown   bool
	prevProbes int
	prevDepth  Depth
	// aboveStreak counts consecutive raw samples above the band: spending
	// recall waits for the streak, not one noisy average, so a controller
	// at rest does not rock between two settings.
	aboveStreak int
	// reranked is what the bound-pruned pass read since the last move.
	reranked int64
}

// New returns a Controller with no tuned state.
func New(cfg Config) *Controller {
	cfg.Target = cmp.Or(cfg.Target, DefaultTarget)
	cfg.SampleRate = cmp.Or(cfg.SampleRate, DefaultSampleRate)
	cfg.MinInterval = cmp.Or(cfg.MinInterval, DefaultMinInterval)
	cfg.MoveSamples = cmp.Or(cfg.MoveSamples, DefaultMoveSamples)
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Controller{cfg: cfg, tune: map[string]*tuning{}, queries: map[string]uint64{}, attempt: map[string]time.Time{}}
}

// Config is the policy with its defaults filled in.
func (c *Controller) Config() Config { return c.cfg }

// On reports whether the controller samples and moves knobs.
func (c *Controller) On() bool { return c.cfg.Target > 0 && c.cfg.Target < 1 }

// Moves is how many knob moves the controller has made.
func (c *Controller) Moves() int64 { return c.moves.Load() }

// Offer counts one live query of index and reports whether it is the one in
// SampleRate to measure. It is a counter, not a timer: an idle index costs
// nothing and a busy one is sampled in proportion.
func (c *Controller) Offer(index string) bool {
	if !c.On() {
		return false
	}
	c.mu.Lock()
	c.queries[index]++
	n := c.queries[index]
	c.mu.Unlock()
	return n%uint64(c.cfg.SampleRate) == 0
}

// Due takes the per-index rate limit. It marks the attempt whether or not
// the measurement then runs: an index that keeps going cold must not be
// re-checked on every offer.
func (c *Controller) Due(index string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.cfg.Now()
	if at, ok := c.attempt[index]; ok && now.Sub(at) < c.cfg.MinInterval {
		return false
	}
	c.attempt[index] = now
	return true
}

// Tuned is what a query of k with no explicit probe count runs at: the
// measured knobs, or the defaults until a measurement says otherwise.
// Filtered keys always get an unbounded depth.
func (c *Controller) Tuned(k Key) Knobs {
	def := Knobs{Depth: c.cfg.Depth}
	if k.Filtered {
		def.Depth = Unbounded
	}
	if !c.On() {
		return def
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.tune[k.Index]
	if t == nil || t.field != k.Field {
		return def
	}
	if k.Filtered {
		if t.filtered == nil {
			return def
		}
		t = t.filtered
	}
	return Knobs{Probes: t.probes, Depth: t.depth}
}

// Set puts k at the given knobs, as if the controller had moved there: to
// restore state saved from an earlier Controller, or to pin one in a test.
// A filtered key keeps its unbounded depth.
func (c *Controller) Set(k Key, at Knobs) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.state(k)
	t.probes = at.Probes
	if !k.Filtered {
		t.depth = at.Depth
	}
}

// Measure runs one sampled query through the warm gate and both arms, and
// folds the recall into k's state. It reports whether a sample landed. It
// returns ErrCold when the index is not warm, the Source's error when an arm
// fails, and false with no error when the approximate arm answered exactly.
// Measure does not take the rate limit; call Due first.
func (c *Controller) Measure(ctx context.Context, k Key, src Source) (bool, error) {
	total, cached := src.Clusters()
	if total == 0 || float64(cached)/float64(total) < warmShare {
		return false, ErrCold
	}
	ann, exact, err := src.Search(ctx, c.Tuned(k))
	if err != nil {
		return false, err
	}
	if exact {
		return false, nil
	}
	truth, err := src.Exact(ctx)
	if err != nil {
		return false, err
	}
	var reranked int64
	if !k.Filtered {
		reranked = src.Reranked()
	}
	c.observe(k, total, Overlap(ann, truth), reranked)
	return true, nil
}

// Overlap is the share of exact's ids that ann also returned; 1 when exact
// is empty.
func Overlap(ann, exact []string) float64 {
	if len(exact) == 0 {
		return 1
	}
	got := make(map[string]bool, len(ann))
	for _, id := range ann {
		got[id] = true
	}
	found := 0
	for _, id := range exact {
		if got[id] {
			found++
		}
	}
	return float64(found) / float64(len(exact))
}

// State is index's unfiltered state, once a sample has landed.
func (c *Controller) State(index string) (State, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.tune[index]
	if t == nil || t.samples == 0 {
		return State{}, false
	}
	return t.publish(Key{Index: index, Field: t.field}), true
}

// States is every state a sample has landed in, filtered ones included.
func (c *Controller) States() []State {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]State, 0, len(c.tune))
	for index, t := range c.tune {
		if t.samples > 0 {
			out = append(out, t.publish(Key{Index: index, Field: t.field}))
		}
		if f := t.filtered; f != nil && f.samples > 0 {
			out = append(out, f.publish(Key{Index: index, Field: t.field, Filtered: true}))
		}
	}
	return out
}

// Forget drops index's state and counters: a recreated index is a
// different one.
func (c *Controller) Forget(index string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.tune, index)
	delete(c.queries, index)
	delete(c.attempt, index)
}

// state is k's tuning, created at the defaults. The caller holds mu.
func (c *Controller) state(k Key) *tuning {
	t := c.tune[k.Index]
	if t == nil || t.field != k.Field {
		t = &tuning{field: k.Field, depth: c.cfg.Depth}
		c.tune[k.Index] = t
	}
	if !k.Filtered {
		return t
	}
	if t.filtered == nil {
		t.filtered = &tuning{field: k.Field, depth: Unbounded, depthFloor: len(ladder) - 1}
	}
	return t.filtered
}

func (t *tuning) publish(k Key) State {
	probes := t.probes
	if probes == 0 {
		probes = ivfq.DefaultNprobe(t.clusters)
	}
	return State{Key: k, Knobs: Knobs{Probes: probes, Depth: t.depth}, Recall: t.recall, Last: t.last,
		Samples: t.samples, Moves: t.moves, At: t.at}
}

// observe folds one sample into k's average and then moves at most one
// knob. The rules, in the order they are applied:
//
//	r < target - floorMargin  after two samples of the new setting, undo
//	                          the last lowering move and pin the value it
//	                          left as a floor;
//	fewer than MoveSamples    do nothing: one query is not evidence;
//	r < target                raise: depth back toward unbounded first (it
//	                          is the cheaper recall), then probes x2,
//	                          capped at 2 x DefaultNprobe;
//	r > target + band         lower: probes /2 first, floored at MinProbes
//	                          (the probe wave is a cold query's whole
//	                          request bill), then one step of depth;
//	otherwise                 inside the band: nothing.
func (c *Controller) observe(k Key, clusters int, recall float64, reranked int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.state(k)
	t.clusters = clusters
	if t.samples == 0 {
		t.recall = recall
	} else {
		t.recall = (1-ewma)*t.recall + ewma*recall
	}
	t.last = recall
	t.samples++
	t.sinceMove++
	t.reranked += reranked
	t.at = c.cfg.Now()
	target := c.cfg.Target
	if recall > target+band {
		t.aboveStreak++
	} else {
		t.aboveStreak = 0
	}
	switch {
	case t.lastKnob != "" && t.lastDown && t.sinceMove >= floorSamples && t.recall < target-floorMargin:
		// The floor rule. The move that got us here cost more recall than
		// the band allows, so undo it and never take that step again.
		if t.lastKnob == "probes" {
			t.probeFloor = max(t.probeFloor, t.prevProbes)
			t.probes = t.prevProbes
		} else {
			t.depthFloor = max(t.depthFloor, t.prevDepth.index())
			t.depth = t.prevDepth
		}
		t.lastKnob, t.sinceMove, t.moves, t.aboveStreak, t.reranked = "", 0, t.moves+1, 0, 0
		c.moves.Add(1)
	case t.sinceMove < c.cfg.MoveSamples:
	case t.recall < target:
		c.move(t, true)
	case t.recall > target+band && t.aboveStreak >= c.cfg.MoveSamples:
		c.move(t, false)
	}
}

// move applies one adjustment. up buys recall, !up spends it. Exactly one
// knob moves, so the next sample attributes cleanly.
func (c *Controller) move(t *tuning, up bool) {
	probes, ceiling := t.probes, maxProbeFactor*ivfq.DefaultNprobe(t.clusters)
	if probes == 0 {
		probes = ivfq.DefaultNprobe(t.clusters)
	}
	floor := max(MinProbes, t.probeFloor)
	at := t.depth.index()
	// The values a floor breach would revert to. They are recorded only if
	// this call moves something: a call that finds every knob pinned must
	// not overwrite the memory of the move that pinned them, or the revert
	// becomes a no-op that pins the floor it was reverting.
	prevProbes, prevDepth := probes, t.depth
	switch {
	case up && at < len(ladder)-1:
		t.depth = ladder[at+1]
		t.lastKnob, t.lastDown = "depth", false
	case up && probes < ceiling:
		t.probes = min(ceiling, probes*2)
		t.lastKnob, t.lastDown = "probes", false
	case !up && probes > floor:
		t.probes = max(floor, probes/2)
		t.lastKnob, t.lastDown = "probes", true
	case !up && at > t.depthFloor:
		if t.reranked == 0 {
			return // the rerank read nothing at this depth: a shallower one is not a step
		}
		next, ok := shallower(at, t.depthFloor, probes)
		if !ok {
			return
		}
		t.depth = ladder[next]
		t.lastKnob, t.lastDown = "depth", true
	default:
		return // pinned against a floor or a ceiling; nothing to move
	}
	t.prevProbes, t.prevDepth = prevProbes, prevDepth
	t.sinceMove, t.moves, t.aboveStreak, t.reranked = 0, t.moves+1, 0, 0
	c.moves.Add(1)
}

// shallower is the next depth down from at that changes what a query does.
// A depth at or above the probe count is the same as unbounded (the rerank
// cannot visit clusters the probe wave never selected), so those rungs are
// stepped past: a move the query cannot feel is a move the next sample
// cannot attribute.
func shallower(at, floor, probes int) (int, bool) {
	for next := at - 1; next >= floor; next-- {
		if d := ladder[next]; d < 0 || int(d) < probes {
			return next, true
		}
	}
	return at, false
}
