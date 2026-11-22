#include "textflag.h"

// func dotsAVX2FMA(q *float32, block *float32, dims int, rows int, out *float32)
// out[r] = Σ q[i]*block[r*dims+i], each row with go-simd/floats'
// dot32AVX2 sequence: one YMM accumulator fused-multiply-added eight lanes
// at a time, folded high-onto-low and twice horizontally, then a scalar
// fused tail.
//
// Rows go four at a time, each in its own accumulator: one row's FMAs are
// a dependent chain, so a row at a time waited out the FMA latency on
// every step (5,464 centroids of 128 dims took 200 µs). Each row still
// runs exactly its own sequence, so the result is unchanged bit for bit.
TEXT ·dotsAVX2FMA(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ dims+16(FP), CX
	MOVQ rows+24(FP), DX
	MOVQ out+32(FP), R9
	MOVQ CX, R10
	SHLQ $2, R10 // a row's bytes

quad:
	CMPQ DX, $4
	JLT  single
	LEAQ (BX)(R10*1), R11
	LEAQ (R11)(R10*1), R12
	LEAQ (R12)(R10*1), R13
	VXORPS Y0, Y0, Y0
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	XORQ DI, DI

qvloop:
	LEAQ 8(DI), R8
	CMPQ R8, CX
	JGT  qvtail
	VMOVUPS (AX)(DI*4), Y1
	VMOVUPS (BX)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y0
	VMOVUPS (R11)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y3
	VMOVUPS (R12)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y4
	VMOVUPS (R13)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y5
	ADDQ $8, DI
	JMP  qvloop

qvtail:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VEXTRACTF128 $1, Y3, X1
	VADDPS X1, X3, X3
	VHADDPS X3, X3, X3
	VHADDPS X3, X3, X3
	VEXTRACTF128 $1, Y4, X1
	VADDPS X1, X4, X4
	VHADDPS X4, X4, X4
	VHADDPS X4, X4, X4
	VEXTRACTF128 $1, Y5, X1
	VADDPS X1, X5, X5
	VHADDPS X5, X5, X5
	VHADDPS X5, X5, X5

	// The scalar tails, each row's in order.
	MOVQ DI, SI

qs0:
	CMPQ SI, CX
	JGE  qs0done
	VMOVSS (AX)(SI*4), X1
	VMOVSS (BX)(SI*4), X2
	VFMADD231SS X2, X1, X0
	ADDQ $1, SI
	JMP  qs0

qs0done:
	MOVQ DI, SI

qs1:
	CMPQ SI, CX
	JGE  qs1done
	VMOVSS (AX)(SI*4), X1
	VMOVSS (R11)(SI*4), X2
	VFMADD231SS X2, X1, X3
	ADDQ $1, SI
	JMP  qs1

qs1done:
	MOVQ DI, SI

qs2:
	CMPQ SI, CX
	JGE  qs2done
	VMOVSS (AX)(SI*4), X1
	VMOVSS (R12)(SI*4), X2
	VFMADD231SS X2, X1, X4
	ADDQ $1, SI
	JMP  qs2

qs2done:
	MOVQ DI, SI

qs3:
	CMPQ SI, CX
	JGE  qs3done
	VMOVSS (AX)(SI*4), X1
	VMOVSS (R13)(SI*4), X2
	VFMADD231SS X2, X1, X5
	ADDQ $1, SI
	JMP  qs3

qs3done:
	VMOVSS X0, (R9)
	VMOVSS X3, 4(R9)
	VMOVSS X4, 8(R9)
	VMOVSS X5, 12(R9)
	LEAQ (R13)(R10*1), BX
	ADDQ $16, R9
	SUBQ $4, DX
	JMP  quad

single:
	TESTQ DX, DX
	JZ   done

row:
	VXORPS Y0, Y0, Y0
	XORQ DI, DI

vloop:
	LEAQ 8(DI), R8
	CMPQ R8, CX
	JGT  vtail
	VMOVUPS (AX)(DI*4), Y1
	VMOVUPS (BX)(DI*4), Y2
	VFMADD231PS Y2, Y1, Y0
	ADDQ $8, DI
	JMP  vloop

vtail:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

sloop:
	CMPQ DI, CX
	JGE  next
	VMOVSS (AX)(DI*4), X1
	VMOVSS (BX)(DI*4), X2
	VFMADD231SS X2, X1, X0
	ADDQ $1, DI
	JMP  sloop

next:
	VMOVSS X0, (R9)
	LEAQ (BX)(CX*4), BX
	ADDQ $4, R9
	DECQ DX
	JNZ  row

done:
	VZEROUPPER
	RET

// func dots3AVX2FMA(a, b, c, block *float32, dims, rows int, out *float32)
// out[3*r+k] = Σ row_k[i]*block[r*dims+i], dims a multiple of 8. Block rows
// go four at a time against the three rows: twelve accumulators, Y(4k+j)
// for row k and block row j, each fused-multiply-added eight lanes at a
// time and folded high-onto-low and twice horizontally. A lone block row
// runs the same sequence for its three sums.
#define HSUM3(y, x) \
	VEXTRACTF128 $1, y, X15 \
	VADDPS X15, x, x \
	VHADDPS x, x, x \
	VHADDPS x, x, x

TEXT ·dots3AVX2FMA(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ c+16(FP), CX
	MOVQ block+24(FP), SI
	MOVQ dims+32(FP), DX
	MOVQ rows+40(FP), R8
	MOVQ out+48(FP), R9
	SHLQ $2, DX // a row's bytes

d3quad:
	CMPQ R8, $4
	JLT  d3single
	LEAQ (SI)(DX*1), R10
	LEAQ (R10)(DX*1), R11
	LEAQ (R11)(DX*1), R12
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
	XORQ DI, DI

d3qloop:
	VMOVUPS (AX)(DI*1), Y12
	VMOVUPS (BX)(DI*1), Y13
	VMOVUPS (CX)(DI*1), Y14
	VMOVUPS (SI)(DI*1), Y15
	VFMADD231PS Y15, Y12, Y0
	VFMADD231PS Y15, Y13, Y4
	VFMADD231PS Y15, Y14, Y8
	VMOVUPS (R10)(DI*1), Y15
	VFMADD231PS Y15, Y12, Y1
	VFMADD231PS Y15, Y13, Y5
	VFMADD231PS Y15, Y14, Y9
	VMOVUPS (R11)(DI*1), Y15
	VFMADD231PS Y15, Y12, Y2
	VFMADD231PS Y15, Y13, Y6
	VFMADD231PS Y15, Y14, Y10
	VMOVUPS (R12)(DI*1), Y15
	VFMADD231PS Y15, Y12, Y3
	VFMADD231PS Y15, Y13, Y7
	VFMADD231PS Y15, Y14, Y11
	ADDQ $32, DI
	CMPQ DI, DX
	JLT  d3qloop

	HSUM3(Y0, X0)
	HSUM3(Y4, X4)
	HSUM3(Y8, X8)
	HSUM3(Y1, X1)
	HSUM3(Y5, X5)
	HSUM3(Y9, X9)
	HSUM3(Y2, X2)
	HSUM3(Y6, X6)
	HSUM3(Y10, X10)
	HSUM3(Y3, X3)
	HSUM3(Y7, X7)
	HSUM3(Y11, X11)
	VMOVSS X0, (R9)
	VMOVSS X4, 4(R9)
	VMOVSS X8, 8(R9)
	VMOVSS X1, 12(R9)
	VMOVSS X5, 16(R9)
	VMOVSS X9, 20(R9)
	VMOVSS X2, 24(R9)
	VMOVSS X6, 28(R9)
	VMOVSS X10, 32(R9)
	VMOVSS X3, 36(R9)
	VMOVSS X7, 40(R9)
	VMOVSS X11, 44(R9)
	LEAQ (R12)(DX*1), SI
	ADDQ $48, R9
	SUBQ $4, R8
	JMP  d3quad

d3single:
	TESTQ R8, R8
	JZ    d3done
	VXORPS Y0, Y0, Y0
	VXORPS Y4, Y4, Y4
	VXORPS Y8, Y8, Y8
	XORQ DI, DI

d3sloop:
	VMOVUPS (AX)(DI*1), Y12
	VMOVUPS (BX)(DI*1), Y13
	VMOVUPS (CX)(DI*1), Y14
	VMOVUPS (SI)(DI*1), Y15
	VFMADD231PS Y15, Y12, Y0
	VFMADD231PS Y15, Y13, Y4
	VFMADD231PS Y15, Y14, Y8
	ADDQ $32, DI
	CMPQ DI, DX
	JLT  d3sloop

	HSUM3(Y0, X0)
	HSUM3(Y4, X4)
	HSUM3(Y8, X8)
	VMOVSS X0, (R9)
	VMOVSS X4, 4(R9)
	VMOVSS X8, 8(R9)
	ADDQ DX, SI
	ADDQ $12, R9
	DECQ R8
	JMP  d3single

d3done:
	VZEROUPPER
	RET
