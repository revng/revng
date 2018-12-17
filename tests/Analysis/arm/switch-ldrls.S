@
@ This file is distributed under the MIT License. See LICENSE.md for details.
@

    .syntax unified
    .global _start
_start:
    cmp     ip, #3
    ldrls   pc, [pc, ip, lsl #2]
    b       end
no:
    .word one
    .word two
    .word three
    .word end
one:
    bx      lr
two:
    bx      lr
three:
    bx      lr
end:
    bx      lr
