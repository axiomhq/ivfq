package bench

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// writeBin writes a Big ANN file: rows, dims, then values.

// TestPercentile: nearest rank on the sorted sample.

func TestPercentile(t *testing.T) {
	ds := make([]time.Duration, 100)
	for i := range ds {
		ds[len(ds)-1-i] = time.Duration(i+1) * time.Millisecond // unsorted on purpose
	}
	for _, tc := range []struct {
		p    float64
		want time.Duration
	}{
		{0.50, 50 * time.Millisecond},
		{0.95, 95 * time.Millisecond},
		{0.99, 99 * time.Millisecond},
	} {
		if got := Percentile(ds, tc.p); got != tc.want {
			t.Errorf("Percentile(%v) = %v, want %v", tc.p, got, tc.want)
		}
	}
	if got := Percentile(nil, 0.5); got != 0 {
		t.Errorf("Percentile(nil) = %v, want 0", got)
	}
}

func TestSummary(t *testing.T) {
	ds := []time.Duration{3 * time.Millisecond, time.Millisecond, 2 * time.Millisecond}
	s := Summary("q", ds)
	for _, want := range []string{"n=3", "p50=2ms", "p99=2ms", "mean=2ms"} {
		if !strings.Contains(s, want) {
			t.Fatalf("%q lacks %q", s, want)
		}
	}
	if ds[0] != 3*time.Millisecond {
		t.Fatal("Summary sorted its input")
	}
}

func TestMemory(t *testing.T) {
	if runtime.GOOS == "linux" && RSSBytes() == 0 {
		t.Fatal("RSSBytes is 0 on linux")
	}
	if MiB(3<<20) != "3 MiB" {
		t.Fatalf("MiB = %q", MiB(3<<20))
	}
	m := StartMemPeak()
	time.Sleep(1100 * time.Millisecond)
	m.Close()
	if m.Sys.Load() == 0 || m.Heap.Load() == 0 {
		t.Fatalf("no sample: Sys %d Heap %d", m.Sys.Load(), m.Heap.Load())
	}
	if runtime.GOOS == "linux" && m.RSS.Load() == 0 {
		t.Fatal("no RSS sample on linux")
	}
}

func TestProfileTo(t *testing.T) {
	dir := t.TempDir()
	sentinel := errors.New("fn")
	if err := ProfileTo(dir, "cpu.pprof", func() error { return sentinel }); err != sentinel {
		t.Fatalf("ProfileTo returned %v, want fn's error", err)
	}
	if st, err := os.Stat(filepath.Join(dir, "cpu.pprof")); err != nil || st.Size() == 0 {
		t.Fatalf("profile not written: %v", err)
	}
	ran := false
	if err := ProfileTo("", "x", func() error { ran = true; return nil }); err != nil || !ran {
		t.Fatal("dir \"\" must run fn bare")
	}
	if runtime.GOOS != "windows" && CPUSeconds() <= 0 {
		t.Fatal("CPUSeconds is 0")
	}
}
