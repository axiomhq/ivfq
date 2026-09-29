package bench

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
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
