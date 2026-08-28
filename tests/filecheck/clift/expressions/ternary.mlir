//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>
!int32_t$const = !clift.const<!clift.int<signed 4>>

%m = clift.imm 0 : !int32_t
%b = clift.test %m : !int32_t
%c = clift.undef : !int32_t$const

clift.ternary %b, %m, %m : !int32_t

// The arguments may have different qualification:
clift.ternary %b, %m, %c : (!int32_t, !int32_t$const)

// Despite two const arguments, the result is non-const:
%t = clift.ternary %b, %c, %c : !int32_t$const

clift.neg %t : !int32_t
