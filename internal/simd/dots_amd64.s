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
