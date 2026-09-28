package ivfq

// Number is every element type a Vector may hold.
type Number interface {
	~float32 | ~float64 | ~int8 | ~int16 | ~int32 | ~uint8 | ~uint16
}

// Vector is a dense row. Methods type-switch on the instantiation: float32
// accumulates in float32 (the SIMD kernels in kernels.go), integers in int32.
type Vector[T Number] []T

// Rows is a late-interaction matrix: one Vector per token.
type Rows[T Number] []Vector[T]

// Sparse is a coordinate-sparse vector. Index is sorted and unique. This is
// the int-indexed form; SparseVector is the string-keyed one.
type Sparse[T Number] struct {
	Index []uint32
	Value []T
}

// Metric names a Distance.
type Metric int

const (
	L2 Metric = iota
	Cosine
	InnerProduct
)
