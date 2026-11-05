package bench

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

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
		{"t.i8bin", []int8{0, 1, 2, 3, 4, 5, -6, -128}},
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
		last := []float32{250, 255}
		if c.name == "t.i8bin" {
			last = []float32{-6, -128}
		}
		if got, err := b.Next(3); err != nil || !reflect.DeepEqual(got, [][]float32{last}) {
			t.Fatalf("%s: last batch %v, %v", c.name, got, err)
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
	// Ending on a row boundary is still a truncation, not the end.
	b2, err := OpenBin(writeBin(t, "short2.u8bin", 5, 2, []uint8{1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	defer b2.Close()
	if got, err := b2.Next(5); !errors.Is(err, io.ErrUnexpectedEOF) || len(got) != 2 {
		t.Fatalf("file truncated at a row boundary: %v, %v", got, err)
	}
	if _, err := OpenBin(writeBin(t, "t.bin", 1, 2, []float32{1, 2})); err == nil {
		t.Fatal("unknown suffix accepted")
	}
	if _, err := OpenBin(writeBin(t, "zero.fbin", 0, 2, []float32{})); err == nil {
		t.Fatal("zero-row header accepted")
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
	// A header claiming 2^31-1 queries is rejected by the file size before
	// anything is allocated.
	if _, err := ReadGroundTruth(writeBin(t, "huge.ibin", math.MaxInt32, 100, []int32{})); err == nil {
		t.Fatal("ground truth larger than its file: want error")
	}
}

// A header whose row count does not fit int32 is rejected, so no reader
// sees a negative count on a 32-bit build.
func TestHugeRowCountRejected(t *testing.T) {
	if _, err := OpenBin(writeBin(t, "huge.u8bin", 1<<31, 2, []uint8{})); err == nil {
		t.Fatal("2^31-row header accepted")
	}
	if _, err := ReadGroundTruth(writeBin(t, "huge31.ibin", 1<<31, 1, []int32{})); err == nil {
		t.Fatal("2^31-query header accepted")
	}
}
