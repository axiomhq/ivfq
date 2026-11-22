#include "textflag.h"

DATA half8<>+0(SB)/8, $0.5
GLOBL half8<>(SB), RODATA|NOPTR, $8
DATA one8<>+0(SB)/8, $1.0
GLOBL one8<>(SB), RODATA|NOPTR, $8
DATA absmask8<>+0(SB)/8, $0x7fffffffffffffff
GLOBL absmask8<>(SB), RODATA|NOPTR, $8
// VPSHUFB mask taking the low byte of each int32 lane to bytes 0..3.
DATA pack8<>+0(SB)/4, $0x0c080400
DATA pack8<>+4(SB)/4, $0x80808080
DATA pack8<>+8(SB)/4, $0x80808080
DATA pack8<>+12(SB)/4, $0x80808080
GLOBL pack8<>(SB), RODATA|NOPTR, $16

// func blockSumSq(block *float32, dims int, sums *float64)
TEXT ·blockSumSq(SB), NOSPLIT, $0-24
	MOVQ block+0(FP), SI
	MOVQ dims+8(FP), CX
	MOVQ sums+16(FP), DI
	VMOVUPD (DI), Y4
	VMOVUPD 32(DI), Y5
	VMOVUPD 64(DI), Y6
	VMOVUPD 96(DI), Y7

loop:
	VCVTPS2PD (SI), Y0
	VCVTPS2PD 16(SI), Y1
	VCVTPS2PD 32(SI), Y2
	VCVTPS2PD 48(SI), Y3
	VMULPD Y0, Y0, Y0
	VMULPD Y1, Y1, Y1
	VMULPD Y2, Y2, Y2
	VMULPD Y3, Y3, Y3
	VADDPD Y0, Y4, Y4
	VADDPD Y1, Y5, Y5
	VADDPD Y2, Y6, Y6
	VADDPD Y3, Y7, Y7
	ADDQ $64, SI
	DECQ CX
	JNZ  loop

	VMOVUPD Y4, (DI)
	VMOVUPD Y5, 32(DI)
	VMOVUPD Y6, 64(DI)
	VMOVUPD Y7, 96(DI)
	VZEROUPPER
	RET

// func blockBits(block *float32, n int, abs *float64, out *uint32)
TEXT ·blockBits(SB), NOSPLIT, $0-32
	MOVQ block+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ abs+16(FP), DI
	MOVQ out+24(FP), R8
	VMOVUPD (DI), Y4
	VMOVUPD 32(DI), Y5
	VMOVUPD 64(DI), Y6
	VMOVUPD 96(DI), Y7
	VBROADCASTSD absmask8<>(SB), Y15
	VPXOR Y14, Y14, Y14 // bits, rows 0-7
	VPXOR Y13, Y13, Y13 // bits, rows 8-15
	VXORPS Y12, Y12, Y12 // zero
	XORQ AX, AX

loop:
	VCVTPS2PD (SI), Y0
	VCVTPS2PD 16(SI), Y1
	VCVTPS2PD 32(SI), Y2
	VCVTPS2PD 48(SI), Y3
	VANDPD Y15, Y0, Y0
	VANDPD Y15, Y1, Y1
	VANDPD Y15, Y2, Y2
	VANDPD Y15, Y3, Y3
	VADDPD Y0, Y4, Y4
	VADDPD Y1, Y5, Y5
	VADDPD Y2, Y6, Y6
	VADDPD Y3, Y7, Y7
	VMOVUPS (SI), Y8
	VMOVUPS 32(SI), Y9
	VCMPPS $0x1d, Y12, Y8, Y8 // >= 0, false for NaN
	VCMPPS $0x1d, Y12, Y9, Y9
	VPSRLD $31, Y8, Y8
	VPSRLD $31, Y9, Y9
	VMOVQ AX, X10
	VPSLLD X10, Y8, Y8
	VPSLLD X10, Y9, Y9
	VPOR Y8, Y14, Y14
	VPOR Y9, Y13, Y13
	ADDQ $64, SI
	INCQ AX
	CMPQ AX, CX
	JLT  loop

	VMOVUPD Y4, (DI)
	VMOVUPD Y5, 32(DI)
	VMOVUPD Y6, 64(DI)
	VMOVUPD Y7, 96(DI)
	VMOVDQU Y14, (R8)
	VMOVDQU Y13, 32(R8)
	VZEROUPPER
	RET

// func minMaxScaled(x *float32, n int, mul float64) (lo, hi float64)
TEXT ·minMaxScaled(SB), NOSPLIT, $0-40
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSD mul+16(FP), Y4
	VCVTPS2PD (SI), Y0
	VMULPD Y4, Y0, Y0
	VMOVAPD Y0, Y1
	ADDQ $16, SI
	SUBQ $4, CX
	JZ   reduce

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y4, Y2, Y2
	VMINPD Y2, Y0, Y0
	VMAXPD Y2, Y1, Y1
	ADDQ $16, SI
	SUBQ $4, CX
	JNZ  loop

reduce:
	VEXTRACTF128 $1, Y0, X2
	VMINPD X2, X0, X0
	VPERMILPD $1, X0, X2
	VMINSD X2, X0, X0
	VEXTRACTF128 $1, Y1, X3
	VMAXPD X3, X1, X1
	VPERMILPD $1, X1, X3
	VMAXSD X3, X1, X1
	VMOVSD X0, lo+24(FP)
	VMOVSD X1, hi+32(FP)
	VZEROUPPER
	RET

// func sumSq(x *float32, n int) float64
TEXT ·sumSq(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $2, CX
	VXORPD Y0, Y0, Y0

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y2, Y2, Y2
	VADDPD Y2, Y0, Y0
	ADDQ $16, SI
	DECQ CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VPERMILPD $1, X0, X1
	VADDSD X1, X0, X0
	VMOVSD X0, ret+16(FP)
	VZEROUPPER
	RET

// func quantCodes(codes *byte, x *float32, n int, mul, lo, delta, levels float64) float64
TEXT ·quantCodes(SB), NOSPLIT, $0-64
	MOVQ codes+0(FP), DI
	MOVQ x+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $2, CX
	VBROADCASTSD mul+24(FP), Y4
	VBROADCASTSD lo+32(FP), Y5
	VBROADCASTSD delta+40(FP), Y6
	VBROADCASTSD levels+48(FP), Y7
	VBROADCASTSD half8<>(SB), Y10
	VBROADCASTSD one8<>(SB), Y11
	VXORPD Y12, Y12, Y12
	VXORPD Y0, Y0, Y0
	VMOVDQU pack8<>(SB), X15

loop:
	VCVTPS2PD (SI), Y2
	VMULPD Y4, Y2, Y2     // x
	VSUBPD Y5, Y2, Y3     // x - lo
	VDIVPD Y6, Y3, Y3     // y
	VROUNDPD $3, Y3, Y8   // t = trunc(y)
	VSUBPD Y8, Y3, Y9     // y - t
	VCMPPD $0x1d, Y10, Y9, Y9
	VANDPD Y11, Y9, Y9
	VADDPD Y9, Y8, Y8     // t++ when y-t >= 0.5
	VMAXPD Y12, Y8, Y8
	VMINPD Y7, Y8, Y8
	VMULPD Y6, Y8, Y13    // delta*t
	VADDPD Y13, Y5, Y13   // lo + delta*t
	VSUBPD Y2, Y13, Y13   // - x
	VMULPD Y13, Y13, Y13
	VADDPD Y13, Y0, Y0
	VCVTTPD2DQY Y8, X14
	VPSHUFB X15, X14, X14
	VMOVD X14, (DI)
	ADDQ $16, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VPERMILPD $1, X0, X1
	VADDSD X1, X0, X0
	VMOVSD X0, ret+56(FP)
	VZEROUPPER
	RET

DATA sfNegZero<>+0(SB)/8, $0x8000000000000000
GLOBL sfNegZero<>(SB), RODATA|NOPTR, $8
DATA sfOne<>+0(SB)/8, $0x3ff0000000000000
GLOBL sfOne<>(SB), RODATA|NOPTR, $8
DATA sfHalf<>+0(SB)/8, $0x3fe0000000000000
GLOBL sfHalf<>(SB), RODATA|NOPTR, $8
DATA sfNegOne<>+0(SB)/8, $0xbff0000000000000
GLOBL sfNegOne<>(SB), RODATA|NOPTR, $8

// func scoreFast4(n int, sums, ones *uint16, norms, aligns, scores *float32, k *float64, cos int)
// Four rows a step in float64 lanes: the Go loop's operations in its order
// (a doubling is x+x, a halving x*0.5, both exact), with both branches of
// its conditions computed and blended. VMAXPD takes the square second, so
// a zero distance is +0 as Go's max gives.
TEXT ·scoreFast4(SB), NOSPLIT, $0-64
	MOVQ n+0(FP), CX
	MOVQ sums+8(FP), SI
	MOVQ ones+16(FP), R8
	MOVQ norms+24(FP), R9
	MOVQ aligns+32(FP), R10
	MOVQ scores+40(FP), DI
	MOVQ k+48(FP), DX
	MOVQ cos+56(FP), R11
	VBROADCASTSD 0(DX), Y8   // delta
	VBROADCASTSD 8(DX), Y9   // low
	VBROADCASTSD 16(DX), Y10 // tail
	VBROADCASTSD 24(DX), Y11 // scale
	VBROADCASTSD 32(DX), Y12 // qn
	VBROADCASTSD 40(DX), Y13 // q2
	VXORPD Y14, Y14, Y14
	SHRQ $2, CX
	JZ   sfdone

sfloop:
	VPMOVZXWD (SI), X0
	VCVTDQ2PD X0, Y0  // sums
	VPMOVZXWD (R8), X1
	VCVTDQ2PD X1, Y1  // ones
	VCVTPS2PD (R9), Y2  // norm
	VCVTPS2PD (R10), Y3 // align
	VMULPD Y8, Y0, Y0  // delta*sums
	VMULPD Y9, Y1, Y1  // low*ones
	VADDPD Y1, Y0, Y0
	VADDPD Y0, Y0, Y0  // 2*(...)
	VSUBPD Y10, Y0, Y0 // - tail
	VMULPD Y11, Y0, Y0 // * scale
	VDIVPD Y3, Y0, Y0  // / align: ip
	VADDPD Y2, Y2, Y4  // 2*norm
	VMULPD Y12, Y4, Y4 // *qn
	VMULPD Y0, Y4, Y4  // *ip
	VMULPD Y2, Y2, Y5  // norm*norm
	VADDPD Y13, Y5, Y5 // + q2
	VSUBPD Y4, Y5, Y6  // - 2*norm*qn*ip
	VCMPPD $0x1e, Y14, Y2, Y7  // norm > 0
	VCMPPD $0x1e, Y14, Y3, Y15 // align > 0
	VANDPD Y15, Y7, Y7
	VBLENDVPD Y7, Y6, Y5, Y5
	VSUBPD Y12, Y2, Y6 // gap
	VMULPD Y6, Y6, Y6
	VMAXPD Y6, Y5, Y5  // max(dist2, gap*gap)
	TESTQ R11, R11
	JNZ  sfcos
	VBROADCASTSD sfNegZero<>(SB), Y15
	VXORPD Y15, Y5, Y5 // -dist2
	JMP  sfstore

sfcos:
	VBROADCASTSD sfHalf<>(SB), Y15
	VMULPD Y15, Y5, Y5 // dist2/2
	VBROADCASTSD sfOne<>(SB), Y15
	VSUBPD Y5, Y15, Y5 // 1 - dist2/2
	VBROADCASTSD sfNegOne<>(SB), Y15
	VMAXPD Y15, Y5, Y5 // max(-1, x)
	VBROADCASTSD sfOne<>(SB), Y15
	VMINPD Y15, Y5, Y5 // min(1, x)

sfstore:
	VBROADCASTSD sfNegOne<>(SB), Y15
	VCMPPD $0x00, Y15, Y3, Y7 // align == -1: a zero row
	VBLENDVPD Y7, Y14, Y5, Y5
	VCVTPD2PSY Y5, X5
	VMOVUPS X5, (DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, DI
	DECQ CX
	JNZ  sfloop

sfdone:
	VZEROUPPER
	RET
