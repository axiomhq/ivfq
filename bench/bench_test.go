package bench

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestReadFvecs(t *testing.T) {
	// two vectors, dims=3, little-endian: [d][f f f][d][f f f]
	var buf bytes.Buffer
	for _, v := range [][]float32{{1, 2, 3}, {4, 5, 6}} {
		binary.Write(&buf, binary.LittleEndian, int32(3))
		binary.Write(&buf, binary.LittleEndian, v)
	}
	p := filepath.Join(t.TempDir(), "t.fvecs")
	os.WriteFile(p, buf.Bytes(), 0o644)
	got, err := ReadFvecs(p)
	if err != nil || len(got) != 2 || !reflect.DeepEqual(got[1], []float32{4, 5, 6}) {
		t.Fatalf("got %v err %v", got, err)
	}
	// truncated trailing record must error, not silently succeed
	os.WriteFile(p, buf.Bytes()[:len(buf.Bytes())-4], 0o644)
	if _, err := ReadFvecs(p); err == nil {
		t.Fatal("truncated file: want error")
	}
}

func TestReadIvecs(t *testing.T) {
	var buf bytes.Buffer
	for _, v := range [][]int32{{7, 8}, {9, 10}} {
		binary.Write(&buf, binary.LittleEndian, int32(2))
		binary.Write(&buf, binary.LittleEndian, v)
	}
	p := filepath.Join(t.TempDir(), "t.ivecs")
	os.WriteFile(p, buf.Bytes(), 0o644)
	if got, err := ReadIvecs(p); err != nil || !reflect.DeepEqual(got, [][]int32{{7, 8}, {9, 10}}) {
		t.Fatalf("got %v err %v", got, err)
	}
	os.WriteFile(p, buf.Bytes()[:len(buf.Bytes())-4], 0o644)
	if _, err := ReadIvecs(p); err == nil {
		t.Fatal("truncated file: want error")
	}
}

// writeBin writes a Big ANN file: rows, dims, then values.
func writeBin(t *testing.T, name string, rows, dims uint32, values any) string {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, [2]uint32{rows, dims})
	binary.Write(&buf, binary.LittleEndian, values)
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBinReader(t *testing.T) {
	for _, c := range []struct {
		name   string
		values any
	}{
		{"t.u8bin", []uint8{0, 1, 2, 3, 4, 5, 250, 255}},
		{"t.fbin", []float32{0, 1, 2, 3, 4, 5, 250, 255}},
	} {
		b, err := OpenBin(writeBin(t, c.name, 4, 2, c.values))
		if err != nil {
			t.Fatal(err)
		}
		if b.Rows() != 4 || b.Dims() != 2 {
			t.Fatalf("%s: header %d×%d", c.name, b.Rows(), b.Dims())
		}
		got, err := b.Next(3)
		if err != nil || !reflect.DeepEqual(got, [][]float32{{0, 1}, {2, 3}, {4, 5}}) {
			t.Fatalf("%s: first batch %v, %v", c.name, got, err)
		}
		if err := b.Skip(3); err != nil {
			t.Fatal(err)
		}
		if got, err := b.Next(3); err != nil || !reflect.DeepEqual(got, [][]float32{{250, 255}}) {
			t.Fatalf("%s: after skip %v, %v", c.name, got, err)
		}
		if _, err := b.Next(3); err != io.EOF {
			t.Fatalf("%s: end %v, want io.EOF", c.name, err)
		}
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
	// A header that promises more rows than the file holds fails the read.
	b, err := OpenBin(writeBin(t, "short.u8bin", 5, 2, []uint8{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if got, err := b.Next(5); err == nil || errors.Is(err, io.EOF) || len(got) != 1 {
		t.Fatalf("truncated file: %v, %v", got, err)
	}
	if _, err := OpenBin(writeBin(t, "zero.fbin", 0, 2, []float32{})); err == nil {
		t.Fatal("zero-row header accepted")
	}
}

func TestU8ToI8(t *testing.T) {
	vecs := [][]float32{{0, 128, 255}}
	U8ToI8(vecs)
	if !reflect.DeepEqual(vecs, [][]float32{{-128, 0, 127}}) {
		t.Fatalf("got %v", vecs)
	}
}

func TestReadGroundTruth(t *testing.T) {
	// 2 queries × 3 ids, then 2×3 distances the reader ignores.
	p := writeBin(t, "gt.ibin", 2, 3, []int32{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0})
	if got, err := ReadGroundTruth(p); err != nil || !reflect.DeepEqual(got, [][]int32{{1, 2, 3}, {4, 5, 6}}) {
		t.Fatalf("got %v err %v", got, err)
	}
	if _, err := ReadGroundTruth(writeBin(t, "short.ibin", 3, 3, []int32{1, 2, 3})); err == nil {
		t.Fatal("truncated ground truth: want error")
	}
}

func TestBlobs(t *testing.T) {
	vecs, centers := Blobs(7, 10, 4, 3)
	if len(vecs) != 10 || len(centers) != 3 || len(vecs[0]) != 4 {
		t.Fatalf("shape %d vecs, %d centers", len(vecs), len(centers))
	}
	again, _ := Blobs(7, 10, 4, 3)
	if !reflect.DeepEqual(vecs, again) {
		t.Fatal("same seed, different vectors")
	}
	// The RNG sequence: all centers first, then each vector's noise.
	rng := rand.New(rand.NewSource(7))
	for range 3 * 4 {
		rng.NormFloat64()
	}
	if want := centers[0][0] + float32(rng.NormFloat64()); vecs[0][0] != want {
		t.Fatalf("vecs[0][0] = %v, want %v", vecs[0][0], want)
	}
}

func TestRecallAtK(t *testing.T) {
	truth := [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	got := [][]int32{{3, 1, 9, 4}, {5}}
	// Query 0: {3, 1} of truth[:2] = {1, 2} → 1 of 2. Query 1: {5} → 1 of 2.
	if r := RecallAtK(got, truth, 2); r != 0.5 {
		t.Fatalf("recall@2 = %v, want 0.5", r)
	}
	// k past both rows: query 0 finds 1, 3, 4 of 4; query 1 finds 5.
	if r := RecallAtK(got, truth, 4); r != (3.0/4+1.0/4)/2 {
		t.Fatalf("recall@4 = %v", r)
	}
	if r := RecallAtK(nil, truth, 10); r != 0 {
		t.Fatalf("no queries: %v", r)
	}
}

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
