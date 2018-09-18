#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call func1
    call func2
    ret

# In this situation we have a return value that is used in one call site and is
# ignored in another one. Therefore it's not Dead, in general.

func1:
    call callee
    mov rax,42
    ret

func2:
    call callee
    mov rbx,rax
    ret

callee:
    mov rax,41
    ret
