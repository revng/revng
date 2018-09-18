#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    add rsp, -8
    call dsaof
    add rsp, 8
    ret

# [rsp+8] is Dead. It's not NoOrDead, since accessing a stack argument,
# automatically implies it actually is an argument.
dsaof:
    mov QWORD PTR [rsp+8],42
    ret
