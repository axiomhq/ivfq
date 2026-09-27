// Package recall tunes an IVF index's two query knobs, the probe count and
// the rerank depth, from a sample of live queries.
//
// Every SampleRate'th query of an index is offered (Offer). A measurement
// (Measure) re-runs it twice on one view of the index: once at the index's
// current Knobs, once exactly. The overlap of the two id lists is recall@k.
// The Controller keeps an exponentially weighted average of it inside
// [Target, Target+0.03] by moving one knob at a time:
//
//   - below the target it buys recall: the rerank depth back toward
//     Unbounded first, then the probe count doubled, up to twice
//     ivfq.DefaultNprobe;
//   - above the band it spends it: the probe count halved first, never
//     below MinProbes, then one step of rerank depth;
//   - two samples more than 0.02 under the target undo the last lowering
//     move and pin the value it came from as a floor for good.
//
// The Controller does no I/O and runs no goroutines. The index is behind a
// Source, bound to one sampled query on one consistent view; the caller
// decides where and when a measurement runs. Tuned state is memory only: a
// new Controller starts from the defaults.
package recall
