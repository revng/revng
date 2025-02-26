//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>

%m = clift.imm 0 : !int32_t
%c = clift.undef : !int32_t$const

clift.ternary %m, %m, %m : (!int32_t, !int32_t)

// The arguments may have different qualification:
clift.ternary %m, %m, %c : (!int32_t, !int32_t, !int32_t$const)

// Despite two const arguments, the result is non-const:
%t = clift.ternary %c, %c, %c : (!int32_t$const, !int32_t$const)
clift.neg %t : !int32_t
