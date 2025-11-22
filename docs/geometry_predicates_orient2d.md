# geometry_predicates::orient2d assembly

- crate: `geometry-predicates v0.3.0` (vendored at `vendor/geometry-predicates`)
- toolchain: `rustc 1.90.0 (1159e78c4 2025-09-14)` on `aarch64-apple-darwin`
- command: `cargo asm geometry_predicates::predicates::orient2d --lib`
- note: the function is annotated `#[inline]`, so temporarily switch it to `#[inline(never)]` before running `cargo asm` and then revert the change once the dump is captured.

### Assembly (release profile)

```asm
	.globl	geometry_predicates::predicates::orient2d
	.p2align	2
geometry_predicates::predicates::orient2d:
Lfunc_begin2:
	.cfi_startproc
	stp x20, x19, [sp, #-32]!
	.cfi_def_cfa_offset 32
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_remember_state
	sub sp, sp, #560
	ldr q4, [x2]
	ext.16b v3, v4, v4, #8
	ldr q7, [x0]
	fsub.2d v1, v7, v4
	ldr q0, [x1]
	ext.16b v5, v0, v0, #8
	fsub.2d v2, v5, v3
	fmul.2d v19, v1, v2
	mov d18, v19[1]
	fsub d0, d19, d18
	fcmp d19, #0.0
	b.le LBB2_3
	fcmp d18, #0.0
	b.ls LBB2_14
	fadd d17, d19, d18
	b LBB2_6
LBB2_3:
	b.pl LBB2_14
	fcmp d18, #0.0
	b.ge LBB2_14
	fneg d6, d19
	fsub d17, d6, d18
LBB2_6:
	mov x8, #4
	movk x8, #15544, lsl #48
	fmov d6, x8
	fmul d6, d17, d6
	fneg d16, d0
	fcmp d0, d6
	fccmp d6, d16, #0, lt
	b.ls LBB2_14
	mov x8, #33554432
	movk x8, #16800, lsl #48
	dup.2d v0, x8
	fmul.2d v6, v1, v0
	fsub.2d v16, v6, v1
	fsub.2d v25, v6, v16
	fsub.2d v24, v1, v25
	fmul.2d v0, v2, v0
	fsub.2d v6, v0, v2
	fsub.2d v16, v0, v6
	fsub.2d v6, v2, v16
	fmul.2d v0, v25, v16
	fsub.2d v0, v19, v0
	fmul.2d v20, v16, v24
	fsub.2d v0, v0, v20
	fmul.2d v20, v25, v6
	fsub.2d v0, v0, v20
	fmul.2d v20, v24, v6
	fsub.2d v0, v20, v0
	mov d20, v0[1]
	fsub d21, d0, d20
	fsub d22, d0, d21
	fadd d23, d21, d22
	fsub d20, d22, d20
	fsub d0, d0, d23
	fadd d0, d20, d0
	fadd d20, d19, d21
	fsub d22, d20, d19
	fsub d23, d20, d22
	fsub d21, d21, d22
	fsub d19, d19, d23
	fadd d19, d21, d19
	fsub d21, d19, d18
	fsub d22, d19, d21
	fadd d23, d21, d22
	fsub d18, d22, d18
	fsub d19, d19, d23
	fadd d18, d18, d19
	fadd d19, d20, d21
	fsub d22, d19, d20
	fsub d23, d19, d22
	fsub d21, d21, d22
	fsub d20, d20, d23
	fadd d20, d21, d20
	stp d0, d18, [sp, #144]
	stp d20, d19, [sp, #160]
	fadd d0, d0, d18
	fadd d0, d0, d20
	fadd d0, d19, d0
	mov x8, #3
	movk x8, #15536, lsl #48
	fmov d18, x8
	fmul d18, d17, d18
	fneg d19, d0
	fcmp d0, d18
	fccmp d18, d19, #0, lt
	b.ls LBB2_14
	fsub.2d v18, v7, v1
	fsub.2d v19, v5, v2
	fadd.2d v20, v1, v18
	fsub.2d v4, v18, v4
	fsub.2d v7, v7, v20
	fadd.2d v7, v4, v7
	fadd.2d v4, v2, v19
	fsub.2d v3, v19, v3
	fsub.2d v4, v5, v4
	fadd.2d v18, v3, v4
	fcmeq.2d v3, v18, #0.0
	fcmeq.2d v4, v7, #0.0
	uzp1.4s v3, v4, v3
	mvn.16b v3, v3
	umaxv.4s s3, v3
	fmov w8, s3
	tbz w8, #0, LBB2_14
	mov x8, #4
	movk x8, #14722, lsl #48
	fmov d3, x8
	fmul d3, d17, d3
	fabs d4, d0
	mov x8, #2
	movk x8, #15544, lsl #48
	fmov d5, x8
	fmul d4, d4, d5
	fmul.2d v17, v1, v18
	fmul.2d v1, v2, v7
	fadd.2d v2, v1, v17
	dup.2d v5, v2[1]
	fadd d3, d4, d3
	fsub.2d v2, v2, v5
	fadd d0, d2, d0
	fneg d2, d0
	fcmp d0, d3
	fccmp d3, d2, #0, lt
	b.ls LBB2_14
	mov x8, #33554432
	movk x8, #16800, lsl #48
	dup.2d v0, x8
	str q0, [sp, #112]
	fmul.2d v0, v7, v0
	fsub.2d v2, v0, v7
	fsub.2d v4, v0, v2
	fsub.2d v3, v7, v4
	fmul.2d v0, v16, v4
	fsub.2d v0, v1, v0
	fmul.2d v2, v16, v3
	fsub.2d v0, v0, v2
	str q4, [sp, #16]
	fmul.2d v2, v6, v4
	fsub.2d v0, v0, v2
	stp q3, q7, [sp, #32]
	fmul.2d v2, v6, v3
	fsub.2d v0, v2, v0
	dup.2d v2, v0[1]
	fsub.2d v2, v0, v2
	fadd.2d v3, v1, v2
	fsub.2d v4, v3, v1
	fsub.2d v5, v3, v4
	fsub.2d v4, v2, v4
	fsub.2d v5, v1, v5
	fadd.2d v4, v4, v5
	dup.2d v5, v1[1]
	fsub.2d v5, v4, v5
	zip1.2d v4, v0, v4
	zip1.2d v6, v2, v5
	fsub.2d v6, v4, v6
	zip1.2d v7, v6, v5
	mov.d v2[1], v6[1]
	mov.d v1[0], v0[1]
	fadd.2d v0, v7, v2
	fsub.2d v1, v6, v1
	fsub.2d v0, v4, v0
	fadd.2d v0, v1, v0
	fadd.2d v1, v3, v5
	fsub.2d v2, v1, v3
	fsub.2d v4, v1, v2
	fsub.2d v2, v5, v2
	fsub.2d v3, v3, v4
	fadd.2d v2, v2, v3
	zip1.2d v1, v2, v1
	stp q0, q1, [sp, #176]
	movi.2d v0, #0000000000000000
	stp q0, q0, [sp, #240]
	stp q0, q0, [sp, #208]
	add x0, sp, #144
	add x2, sp, #176
	add x3, sp, #208
	mov w1, #4
	mov w4, #8
	stp q17, q24, [sp, #80]
	str q25, [sp, #64]
	str q18, [sp, #128]
	bl geometry_predicates::predicates::fast_expansion_sum_zeroelim
	mov x1, x0
	ldp q0, q2, [sp, #112]
	fmul.2d v0, v2, v0
	fsub.2d v1, v0, v2
	fsub.2d v1, v0, v1
	fsub.2d v2, v2, v1
	ldp q4, q7, [sp, #64]
	fmul.2d v0, v4, v1
	fsub.2d v0, v7, v0
	str q1, [sp]
	ldr q3, [sp, #96]
	fmul.2d v1, v3, v1
	fsub.2d v0, v0, v1
	fmul.2d v1, v4, v2
	fsub.2d v0, v0, v1
	str q2, [sp, #112]
	fmul.2d v1, v3, v2
	fsub.2d v0, v1, v0
	dup.2d v1, v0[1]
	fsub.2d v1, v0, v1
	fadd.2d v2, v7, v1
	fsub.2d v3, v2, v7
	fsub.2d v4, v2, v3
	fsub.2d v3, v1, v3
	fsub.2d v4, v7, v4
	dup.2d v5, v7[1]
	fadd.2d v3, v3, v4
	fsub.2d v4, v3, v5
	zip1.2d v3, v0, v3
	zip1.2d v5, v1, v4
	fsub.2d v5, v3, v5
	zip1.2d v6, v5, v4
	mov.d v1[1], v5[1]
	fadd.2d v1, v6, v1
	mov.d v7[0], v0[1]
	fsub.2d v0, v5, v7
	fsub.2d v1, v3, v1
	fadd.2d v0, v0, v1
	fadd.2d v1, v2, v4
	fsub.2d v3, v1, v2
	fsub.2d v5, v1, v3
	fsub.2d v3, v4, v3
	fsub.2d v2, v2, v5
	fadd.2d v2, v3, v2
	zip1.2d v1, v2, v1
	stp q0, q1, [sp, #272]
	movi.2d v0, #0000000000000000
	stp q0, q0, [sp, #368]
	stp q0, q0, [sp, #336]
	stp q0, q0, [sp, #304]
	cmp x0, #9
	b.hs LBB2_15
	add x0, sp, #208
	add x2, sp, #272
	add x3, sp, #304
	mov w4, #12
	bl geometry_predicates::predicates::fast_expansion_sum_zeroelim
	mov x1, x0
	ldp q7, q0, [sp, #32]
	ldp q16, q1, [sp, #112]
	fmul.2d v0, v0, v1
	ldp q5, q3, [sp]
	fmul.2d v1, v3, v5
	fsub.2d v1, v0, v1
	fmul.2d v2, v7, v5
	fsub.2d v1, v1, v2
	fmul.2d v2, v3, v16
	fsub.2d v1, v1, v2
	fmul.2d v2, v7, v16
	fsub.2d v1, v2, v1
	mov d2, v0[1]
	mov d3, v3[1]
	mov d4, v5[1]
	fmul.d d5, d3, v5[1]
	fsub d5, d2, d5
	mov d6, v7[1]
	fmul.d d4, d4, v7[1]
	fsub d4, d5, d4
	fmul.d d3, d3, v16[1]
	fsub d3, d4, d3
	fmul.d d4, d6, v16[1]
	fsub d3, d4, d3
	fsub d4, d1, d3
	fsub d5, d1, d4
	fadd d6, d4, d5
	fsub d3, d5, d3
	fsub d1, d1, d6
	fadd d1, d3, d1
	fadd d3, d0, d4
	fsub d5, d3, d0
	fsub d6, d3, d5
	fsub d4, d4, d5
	fsub d0, d0, d6
	fadd d0, d4, d0
	fsub d4, d0, d2
	fsub d5, d0, d4
	fadd d6, d4, d5
	fsub d2, d5, d2
	fsub d0, d0, d6
	fadd d0, d2, d0
	fadd d2, d3, d4
	fsub d5, d2, d3
	fsub d6, d2, d5
	fsub d4, d4, d5
	fsub d3, d3, d6
	stp d1, d0, [x29, #-176]
	fadd d0, d4, d3
	stp d0, d2, [x29, #-160]
	movi.2d v0, #0000000000000000
	stp q0, q0, [x29, #-48]
	stp q0, q0, [x29, #-80]
	stp q0, q0, [x29, #-112]
	stp q0, q0, [x29, #-144]
	cmp x0, #13
	b.hs LBB2_16
	sub x19, x29, #144
	add x0, sp, #304
	sub x2, x29, #176
	sub x3, x29, #144
	mov w4, #16
	bl geometry_predicates::predicates::fast_expansion_sum_zeroelim
	mov x8, x0
	sub x0, x0, #1
	cmp x8, #16
	b.hi LBB2_17
	ldr d0, [x19, x0, lsl #3]
LBB2_14:
	add sp, sp, #560
	.cfi_def_cfa wsp, 32
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	ret
LBB2_15:
	.cfi_restore_state
Lloh20:
	adrp x2, l_anon.b9b79bb672675096bac3a0f032032370.11@PAGE
Lloh21:
	add x2, x2, l_anon.b9b79bb672675096bac3a0f032032370.11@PAGEOFF
	mov x0, x1
	mov w1, #8
	bl core::slice::index::slice_end_index_len_fail
LBB2_16:
Lloh22:
	adrp x2, l_anon.b9b79bb672675096bac3a0f032032370.12@PAGE
Lloh23:
	add x2, x2, l_anon.b9b79bb672675096bac3a0f032032370.12@PAGEOFF
	mov x0, x1
	mov w1, #12
	bl core::slice::index::slice_end_index_len_fail
LBB2_17:
Lloh24:
	adrp x2, l_anon.b9b79bb672675096bac3a0f032032370.13@PAGE
Lloh25:
	add x2, x2, l_anon.b9b79bb672675096bac3a0f032032370.13@PAGEOFF
	mov w1, #16
	bl core::panicking::panic_bounds_check
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpAdd	Lloh22, Lloh23
	.loh AdrpAdd	Lloh24, Lloh25
```
