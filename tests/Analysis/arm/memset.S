    .syntax unified
    .global _start
_start:
memset:
    mov     r3, r0
    cmp     r2, #8
    bcc     final
    orr     r1, r1, r1, lsl #8
    orr     r1, r1, r1, lsl #16
loop:
    tst     r3, #3
    strbne  r1, [r3], #1
    subne   r2, r2, #1
    bne     loop
    mov     ip, r1
again:
    cmp     r2, #8
    bcc     final
    stmia   r3!, {r1, ip}
    sub     r2, r2, #8
    cmp     r2, #8
    bcc     final
    stmia   r3!, {r1, ip}
    sub     r2, r2, #8
    cmp     r2, #8
    bcc     final
    stmia   r3!, {r1, ip}
    sub     r2, r2, #8
    cmp     r2, #8
    stmiacs r3!, {r1, ip}
    subcs   r2, r2, #8
    bcs     again
final:
    movs    r2, r2
    moveq   pc, lr
    rsb     r2, r2, #7
    add     pc, pc, r2, lsl #2
    nop
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    strb    r1, [r3], #1
    mov     pc, lr
    nop     {0}
