    .syntax unified
    .global _start
_start:
    cmp     ip, #3
    ldrls   pc, [pc, ip, lsl #2]
    b       end
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
