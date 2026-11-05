package bench

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// Big ANN binary formats (.u8bin, .i8bin, .fbin, .ibin): two little-endian
// uint32 (rows, dims), then row-major values.

type binHeader struct{ rows, dims int }

func readBinHeader(f *os.File) (binHeader, error) {
	var h [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &h); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF // an empty file, not an empty corpus
		}
		return binHeader{}, fmt.Errorf("%s: header: %w", f.Name(), err)
	}
	// rows fits int32 so that every row count is a positive int, 32-bit
	// included; Big ANN corpora stop at 1e9.
	if h[0] == 0 || h[0] > math.MaxInt32 || h[1] == 0 || h[1] > 4096 {
		return binHeader{}, fmt.Errorf("%s: implausible header rows=%d dims=%d", f.Name(), h[0], h[1])
	}
	return binHeader{rows: int(h[0]), dims: int(h[1])}, nil
}

// BinReader streams rows of a .u8bin (uint8), .i8bin (int8) or .fbin
// (float32) file as float32 vectors, in bounded batches. A sample cut from
// a corpus keeps the corpus header, so it holds fewer rows than Rows: read
// the rows you need and stop, since reading past its end is a truncation.
type BinReader struct {
	f   *os.File
	r   *bufio.Reader
	h   binHeader
	ext string
	row int
}

// OpenBin opens a Big ANN vector file; its suffix selects the value type,
// and any suffix but .u8bin, .i8bin and .fbin is an error.
func OpenBin(path string) (*BinReader, error) {
	ext := filepath.Ext(path)
	switch ext {
	case ".u8bin", ".i8bin", ".fbin":
	default:
		return nil, fmt.Errorf("%s: not a .u8bin, .i8bin or .fbin file", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	h, err := readBinHeader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &BinReader{f: f, r: bufio.NewReaderSize(f, 8<<20), h: h, ext: ext}, nil
}

// Rows is the row count in the header.
func (b *BinReader) Rows() int { return b.h.rows }

// Dims is the row width in the header.
func (b *BinReader) Dims() int { return b.h.dims }

// Next returns up to n (> 0) rows as float32 vectors, or io.EOF after the
// header's last row. A file that ends early returns the rows read so far and
// an error wrapping io.ErrUnexpectedEOF.
func (b *BinReader) Next(n int) ([][]float32, error) {
	out := make([][]float32, 0, n)
	elem := 1
	if b.ext == ".fbin" {
		elem = 4
	}
	buf := make([]byte, b.h.dims*elem)
	for len(out) < n && b.row < b.h.rows {
		if _, err := io.ReadFull(b.r, buf); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF // the header promised this row
			}
			return out, fmt.Errorf("%s: row %d: %w", b.f.Name(), b.row, err)
		}
		v := make([]float32, b.h.dims)
		switch b.ext {
		case ".u8bin":
			for i, x := range buf {
				v[i] = float32(x)
			}
		case ".i8bin":
			for i, x := range buf {
				v[i] = float32(int8(x))
			}
		default:
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
	// Check the size before allocating what a corrupt header claims.
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if want := 8 + 8*int64(h.rows)*int64(h.dims); st.Size() < want {
		return nil, fmt.Errorf("%s: header rows=%d k=%d needs %d bytes, file has %d", path, h.rows, h.dims, want, st.Size())
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
