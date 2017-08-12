#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call indirectcall
    ret

indirectcall:
    push rbx
    call rbx
    pop rbx
    ret
