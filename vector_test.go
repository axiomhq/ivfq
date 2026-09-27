package ivfq

import "testing"

func TestVectorTable(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		a := Vector[float32]{1, 2, 3}
		b := Vector[float32]{4, 5, 6}
		if a.Dot(b) != kernelDot([]float32(a), []float32(b)) {
			t.Fatalf("Dot %v", a.Dot(b))
		}
		if a.L2(b) != kernelL2([]float32(a), []float32(b)) {
			t.Fatalf("L2 %v", a.L2(b))
		}
		if a.Cosine(b) != kernelCosine([]float32(a), []float32(b)) {
			t.Fatalf("Cosine %v", a.Cosine(b))
		}
		if Distance(InnerProduct, a, b) != a.Dot(b) || Distance(L2, a, b) != a.L2(b) {
			t.Fatal("Distance")
		}
		if n := a.Norm(); n != kernelNorm([]float32(a)) {
			t.Fatalf("Norm %v", n)
		}
		u := a.Normalized()
		if u.Norm() < 0.99 || u.Norm() > 1.01 {
			t.Fatalf("Normalized norm %v", u.Norm())
		}
	})
	t.Run("int8", func(t *testing.T) {
		a := Vector[int8]{1, 2, 3}
		b := Vector[int8]{4, 5, 6}
		if a.Dot(b) != 1*4+2*5+3*6 {
			t.Fatalf("Dot %v", a.Dot(b))
		}
		if a.L2(b) != (1-4)*(1-4)+(2-5)*(2-5)+(3-6)*(3-6) {
			t.Fatalf("L2 %v", a.L2(b))
		}
		if Distance(InnerProduct, a, b) != a.Dot(b) {
			t.Fatal("Distance")
		}
		s := Sparse[int8]{Index: []uint32{0, 2}, Value: []int8{1, 3}}
		o := Sparse[int8]{Index: []uint32{1, 2}, Value: []int8{9, 4}}
		if s.Dot(o) != 12 {
			t.Fatalf("Sparse.Dot %v", s.Dot(o))
		}
		rows := Rows[int8]{{1, 0}, {0, 1}}
		if rows.MaxSim(InnerProduct, Rows[int8]{{1, 0}}) != 1 {
			t.Fatalf("MaxSim %v", rows.MaxSim(InnerProduct, Rows[int8]{{1, 0}}))
		}
	})
}

func BenchmarkDotVectorF32(b *testing.B) {
	a := make(Vector[float32], 128)
	w := make(Vector[float32], 128)
	for i := range a {
		a[i], w[i] = float32(i), float32(i+1)
	}
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += a.Dot(w)
	}
	_ = s
}

func BenchmarkDotKernelF32(b *testing.B) {
	a := make([]float32, 128)
	w := make([]float32, 128)
	for i := range a {
		a[i], w[i] = float32(i), float32(i+1)
	}
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += kernelDot(a, w)
	}
	_ = s
}
