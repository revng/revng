#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

.intel_syntax noprefix
.global _start
_start:
    call wrapper
    ret

wrapper:
    call recursive
    ret

recursive:
    je skip
    call intermediate
skip:
    ret

intermediate:
    call recursive
    ret
