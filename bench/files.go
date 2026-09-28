package bench

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// ReadFvecs parses the texmex fvecs format: repeated records of a
// little-endian int32 dim followed by dim float32s. Errors on d <= 0, d
// exceeding a sanity ceiling, mixed dims across records, or a truncated
// trailing record.
func ReadFvecs(path string) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var vecs [][]float32
	dims := -1
	for {
		var d int32
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("%s: reading dim of record %d: %w", path, len(vecs), err)
		}
		if d <= 0 || d > 4096 {
			return nil, fmt.Errorf("%s: record %d: invalid dim %d", path, len(vecs), d)
		}
		if dims == -1 {
			dims = int(d)
		} else if int(d) != dims {
			return nil, fmt.Errorf("%s: record %d: dim %d != first record's dim %d", path, len(vecs), d, dims)
		}
		v := make([]float32, d)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("%s: record %d: reading %d floats: %w", path, len(vecs), d, err)
		}
		vecs = append(vecs, v)
	}
	return vecs, nil
}

// ReadIvecs reads an ivecs file (int32 count, then that many int32s per
// record): the exact-neighbour ground truth that ships with SIFT1M and GIST.
func ReadIvecs(path string) ([][]int32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var recs [][]int32
	for {
		var d int32
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("%s: reading count of record %d: %w", path, len(recs), err)
		}
		if d <= 0 || d > 4096 {
			return nil, fmt.Errorf("%s: record %d: invalid count %d", path, len(recs), d)
		}
		rec := make([]int32, d)
		if err := binary.Read(r, binary.LittleEndian, rec); err != nil {
			return nil, fmt.Errorf("%s: record %d: %w", path, len(recs), err)
		}
		recs = append(recs, rec)
	}
	return recs, nil
}

// Big ANN binary formats (.u8bin, .fbin, .ibin): two little-endian uint32
// (rows, dims), then row-major values.

type binHeader struct{ rows, dims int }

func readBinHeader(f *os.File) (binHeader, error) {
	var h [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &h); err != nil {
		return binHeader{}, err
	}
	if h[0] == 0 || h[1] == 0 || h[1] > 4096 {
		return binHeader{}, fmt.Errorf("%s: implausible header rows=%d dims=%d", f.Name(), h[0], h[1])
	}
	return binHeader{rows: int(h[0]), dims: int(h[1])}, nil
}

// BinReader streams rows of a .u8bin (one byte per value) or .fbin
// (float32) file as float32 vectors, in bounded batches.
type BinReader struct {
	f    *os.File
	r    *bufio.Reader
	h    binHeader
	elem int
	row  int
}

// OpenBin opens a Big ANN vector file; the .u8bin suffix selects one byte
// per value, anything else float32.
func OpenBin(path string) (*BinReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	h, err := readBinHeader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	elem := 4
	if strings.HasSuffix(path, ".u8bin") {
		elem = 1
	}
	return &BinReader{f: f, r: bufio.NewReaderSize(f, 8<<20), h: h, elem: elem}, nil
}

// Rows is the row count in the header.
func (b *BinReader) Rows() int { return b.h.rows }

// Dims is the row width in the header.
func (b *BinReader) Dims() int { return b.h.dims }

// Skip positions the reader at row rows, counted from the start of the file.
func (b *BinReader) Skip(rows int) error {
	if rows <= 0 {
		return nil
	}
	if _, err := b.f.Seek(int64(8+rows*b.h.dims*b.elem), io.SeekStart); err != nil {
		return err
	}
	b.r.Reset(b.f)
	b.row = rows
	return nil
}

// Next returns up to n rows as float32 vectors, or io.EOF when done.
func (b *BinReader) Next(n int) ([][]float32, error) {
	out := make([][]float32, 0, n)
	buf := make([]byte, b.h.dims*b.elem)
	for len(out) < n && b.row < b.h.rows {
		if _, err := io.ReadFull(b.r, buf); err != nil {
			return out, fmt.Errorf("%s: row %d: %w", b.f.Name(), b.row, err)
		}
		v := make([]float32, b.h.dims)
		if b.elem == 1 {
			for i, x := range buf {
				v[i] = float32(x)
			}
		} else {
			for i := range v {
				v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
			}
		}
		out = append(out, v)
		b.row++
	}
	if len(out) == 0 {
		return nil, io.EOF
	}
	return out, nil
}

// Close closes the file.
func (b *BinReader) Close() error { return b.f.Close() }

// i8Shift is what i8 storage subtracts from every .u8bin value, base and
// query alike, to move 0..255 into i8's -128..127. Moving every point by the
// same vector leaves every l2 distance, and so the ground truth, unchanged;
// cosine distances would change.
const i8Shift = 128

// U8ToI8 shifts vectors read from a .u8bin file into i8's range, in place.
// It preserves l2 distances, not cosine ones.
func U8ToI8(vecs [][]float32) {
	for _, v := range vecs {
		for i := range v {
			v[i] -= i8Shift
		}
	}
}

// ReadGroundTruth reads a Big ANN .ibin ground truth: (queries, k) then
// queries×k int32 ids, then queries×k float32 distances (ignored).
func ReadGroundTruth(path string) ([][]int32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h, err := readBinHeader(f)
	if err != nil {
		return nil, err
	}
	r := bufio.NewReaderSize(f, 4<<20)
	out := make([][]int32, h.rows)
	for i := range out {
		out[i] = make([]int32, h.dims)
		if err := binary.Read(r, binary.LittleEndian, out[i]); err != nil {
			return nil, fmt.Errorf("%s: query %d: %w", path, i, err)
		}
	}
	return out, nil
}
