@
@ This file is distributed under the MIT License. See LICENSE.md for details.
@

    .syntax unified
    .global _start
_start:
    push {lr}
    bl tailcall
    pop {lr}
    bx lr

tailcall:
    push {r0}
    pop {r0}
    bx r0
