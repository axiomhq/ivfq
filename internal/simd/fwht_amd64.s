#include "textflag.h"

// Per-lane signs for the three in-register stages: +1 where the lane is the
// lower of its pair, -1 where it is the upper.
DATA signs8<>+0(SB)/4, $0x3f800000
DATA signs8<>+4(SB)/4, $0xbf800000
DATA signs8<>+8(SB)/4, $0x3f800000
DATA signs8<>+12(SB)/4, $0xbf800000
DATA signs8<>+16(SB)/4, $0x3f800000
DATA signs8<>+20(SB)/4, $0xbf800000
DATA signs8<>+24(SB)/4, $0x3f800000
DATA signs8<>+28(SB)/4, $0xbf800000
DATA signs8<>+32(SB)/4, $0x3f800000
DATA signs8<>+36(SB)/4, $0x3f800000
DATA signs8<>+40(SB)/4, $0xbf800000
DATA signs8<>+44(SB)/4, $0xbf800000
DATA signs8<>+48(SB)/4, $0x3f800000
DATA signs8<>+52(SB)/4, $0x3f800000
DATA signs8<>+56(SB)/4, $0xbf800000
DATA signs8<>+60(SB)/4, $0xbf800000
DATA signs8<>+64(SB)/4, $0x3f800000
DATA signs8<>+68(SB)/4, $0x3f800000
DATA signs8<>+72(SB)/4, $0x3f800000
DATA signs8<>+76(SB)/4, $0x3f800000
DATA signs8<>+80(SB)/4, $0xbf800000
DATA signs8<>+84(SB)/4, $0xbf800000
DATA signs8<>+88(SB)/4, $0xbf800000
DATA signs8<>+92(SB)/4, $0xbf800000
GLOBL signs8<>(SB), RODATA|NOPTR, $96

// func fwht8(x *float32, n int)
// Each run of 8 floats becomes its Hadamard transform: three stages, each
// "neighbour + sign*self" (a+b for the lower lane, a-b for the upper).
TEXT ·fwht8(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX
	SHRQ $3, CX
	JZ   done
	VMOVUPS signs8<>+0(SB), Y8
	VMOVUPS signs8<>+32(SB), Y9
	VMOVUPS signs8<>+64(SB), Y10

loop:
	VMOVUPS (DI), Y0
	VPERMILPS $0xb1, Y0, Y1
	VFMADD231PS Y8, Y0, Y1
	VPERMILPS $0x4e, Y1, Y2
	VFMADD231PS Y9, Y1, Y2
	VPERM2F128 $0x01, Y2, Y2, Y3
	VFMADD231PS Y10, Y2, Y3
	VMOVUPS Y3, (DI)
	ADDQ $32, DI
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET

// func fwhtStages(x *float32, n, h0 int)
TEXT ·fwhtStages(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), R8
	MOVQ h0+16(FP), R10
	MOVQ R8, R12
	SHLQ $2, R12
	ADDQ DI, R12 // end of x

stage:
	CMPQ R10, R8
	JGE  done
	MOVQ R10, R11
	SHLQ $2, R11 // run length in bytes
	MOVQ DI, SI

block:
	CMPQ SI, R12
	JGE  next
	LEAQ (SI)(R11*1), BX
	XORQ DX, DX

lanes:
	VMOVUPS (SI)(DX*1), Y0
	VMOVUPS (BX)(DX*1), Y1
	VADDPS Y1, Y0, Y2
	VSUBPS Y1, Y0, Y3
	VMOVUPS Y2, (SI)(DX*1)
	VMOVUPS Y3, (BX)(DX*1)
	ADDQ $32, DX
	CMPQ DX, R11
	JLT  lanes
	LEAQ (SI)(R11*2), SI
	JMP  block

next:
	SHLQ $1, R10
	JMP  stage

done:
	VZEROUPPER
	RET

// func permScale(dst, src *float32, perm *int32, coef *float32, n int)
// dst[i] = coef[i] * src[perm[i]], one element at a time: VGATHERDPS is
// slower than these loads on CPUs with the gather-data-sampling mitigation.
TEXT ·permScale(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ perm+16(FP), BX
	MOVQ coef+24(FP), DX
	MOVQ n+32(FP), CX
	SHRQ $2, CX
	JZ   done

loop:
	MOVLQSX (BX), AX
	MOVLQSX 4(BX), R8
	MOVLQSX 8(BX), R9
	MOVLQSX 12(BX), R10
	VMOVSS (SI)(AX*4), X0
	VMOVSS (SI)(R8*4), X1
	VMOVSS (SI)(R9*4), X2
	VMOVSS (SI)(R10*4), X3
	VMULSS (DX), X0, X0
	VMULSS 4(DX), X1, X1
	VMULSS 8(DX), X2, X2
	VMULSS 12(DX), X3, X3
	VMOVSS X0, (DI)
	VMOVSS X1, 4(DI)
	VMOVSS X2, 8(DI)
	VMOVSS X3, 12(DI)
	ADDQ $16, BX
	ADDQ $16, DX
	ADDQ $16, DI
	DECQ CX
	JNZ  loop

done:
	RET

// func scale(dst, src, coef *float32, n int)
// dst[i] = coef[i] * src[i], n a multiple of 8.
TEXT ·scale(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ coef+16(FP), DX
	MOVQ n+24(FP), CX
	SHRQ $3, CX
	JZ   done

loop:
	VMOVUPS (SI), Y0
	VMULPS (DX), Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET

// func permScaleRows(dst, src *float32, perm *int32, coef *float32, n int)
TEXT ·permScaleRows(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ perm+16(FP), BX
	MOVQ coef+24(FP), DX
	MOVQ n+32(FP), CX

loop:
	MOVLQSX (BX), AX
	SHLQ $6, AX
	VBROADCASTSS (DX), Y2
	VMULPS (SI)(AX*1), Y2, Y0
	VMULPS 32(SI)(AX*1), Y2, Y1
	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	ADDQ $4, BX
	ADDQ $4, DX
	ADDQ $64, DI
	DECQ CX
	JNZ  loop

	VZEROUPPER
	RET
