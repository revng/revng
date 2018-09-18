#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call push_pop
    ret

push_pop:
    push rax
    pop rdx
    ret
