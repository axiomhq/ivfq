package bench

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Percentile returns the p-quantile of an unsorted duration sample (nearest
// rank on the sorted slice; it sorts ds in place); 0 for none.
func Percentile(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	i := int(p * float64(len(ds)-1))
	return ds[i]
}

// Summary is one line: label, count, p50, p95, p99 and mean, rounded to the
// microsecond. ds is not modified; it must not be empty.
func Summary(label string, ds []time.Duration) string {
	c := append([]time.Duration(nil), ds...)
	var sum time.Duration
	for _, d := range c {
		sum += d
	}
	return fmt.Sprintf("%-22s n=%-4d p50=%-10v p95=%-10v p99=%-10v mean=%v",
		label, len(c), Percentile(c, 0.50).Round(time.Microsecond), Percentile(c, 0.95).Round(time.Microsecond),
		Percentile(c, 0.99).Round(time.Microsecond), (sum / time.Duration(len(c))).Round(time.Microsecond))
}

// RSSBytes is this process's resident set from /proc/self/statm. It is 0
// where there is no /proc (anything but Linux) or on a read failure, which
// loses one sample and never stops a run.
func RSSBytes() uint64 {
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	f := strings.Fields(string(b))
	if len(f) < 2 {
		return 0
	}
	pages, err := strconv.ParseUint(f[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * uint64(os.Getpagesize())
}

// MemPeak samples memory every second and keeps the high-water marks until
// Close: the Go runtime's Sys and HeapInuse, and the process RSS, because at
// scale they diverge — Go memory is what the program holds, RSS is the
// number that decides whether a run of this size fits a machine.
type MemPeak struct {
	Sys, Heap, RSS atomic.Uint64
	stop           chan struct{}
	done           chan struct{}
}

// StartMemPeak starts the 1 Hz sampler.
func StartMemPeak() *MemPeak {
	m := &MemPeak{stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(m.done)
		t := time.NewTicker(time.Second)
		defer t.Stop()
		var ms runtime.MemStats
		for {
			select {
			case <-m.stop:
				return
			case <-t.C:
			}
			runtime.ReadMemStats(&ms)
			for cur := m.Sys.Load(); ms.Sys > cur; cur = m.Sys.Load() {
				if m.Sys.CompareAndSwap(cur, ms.Sys) {
					break
				}
			}
			for cur := m.Heap.Load(); ms.HeapInuse > cur; cur = m.Heap.Load() {
				if m.Heap.CompareAndSwap(cur, ms.HeapInuse) {
					break
				}
			}
			r := RSSBytes()
			for cur := m.RSS.Load(); r > cur; cur = m.RSS.Load() {
				if m.RSS.CompareAndSwap(cur, r) {
					break
				}
			}
		}
	}()
	return m
}

// Close stops the sampler and waits for it; the peaks stay readable.
func (m *MemPeak) Close() { close(m.stop); <-m.done }

// MiB formats a byte count as whole mebibytes.
func MiB(b uint64) string { return fmt.Sprintf("%.0f MiB", float64(b)/(1<<20)) }

// ProfileTo runs fn under a CPU profile written to dir/name; dir "" runs fn
// bare. A profile that cannot be started is reported on stdout and skipped:
// a benchmark must not die because the profile file is unwritable.
func ProfileTo(dir, name string, fn func() error) error {
	if dir == "" {
		return fn()
	}
	f, p := createProfile(dir, name)
	if f == nil {
		return fn()
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Printf("profile %s: %v (skipping)\n", p, err)
		f.Close()
		return fn()
	}
	err := fn()
	pprof.StopCPUProfile()
	f.Close()
	fmt.Printf("cpu profile: %s\n", p)
	return err
}

func createProfile(dir, name string) (*os.File, string) {
	p := filepath.Join(dir, name)
	_ = os.MkdirAll(dir, 0o755)
	f, err := os.Create(p)
	if err != nil {
		fmt.Printf("profile %s: %v (skipping)\n", p, err)
		return nil, p
	}
	return f, p
}
