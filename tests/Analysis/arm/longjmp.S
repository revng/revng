@
@ This file is distributed under the MIT License. See LICENSE.md for details.
@

    .syntax unified
    .global _start
_start:
    push {lr}
    bl longjmp
    pop {lr}
    bx lr

longjmp:
    ble ret
    push {r0}
    bx r0
ret:
    bx lr
