#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    push rax
    push rax
    call usaof
    lea rsp, [rsp+16]
    ret

usaof:
    # This is a used argument on the stack
    mov rbx, QWORD PTR [rsp+8]
    # This is a used argument on the stack too, despite being
    # clobbered. Therefore, it's Dead (not NoOrDead)
    mov QWORD PTR [rsp+16], rbx
    ret
