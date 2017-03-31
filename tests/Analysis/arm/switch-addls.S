@
@ This file is distributed under the MIT License. See LICENSE.md for details.
@

    .syntax unified
    .global _start
_start:
    cmp     r1, #4
    addls   pc, pc, r1, lsl #2
    b       end
    b       end
    b       end
    b       end
    b       end
    b       end
end:
    bx      lr
