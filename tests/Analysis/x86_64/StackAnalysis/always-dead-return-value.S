#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call func
    ret

func:
    call callee
    mov rax,42
    ret

# In this function rax is a return value, but on all the call sites the return
# value is ignored.
callee:
    mov rax,41
    ret
