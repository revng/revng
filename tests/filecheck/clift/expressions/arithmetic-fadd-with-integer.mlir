//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

%i = clift.undef : !int32_t

// CHECK: operand #0 must be floating point type
clift.fadd %i, %i : !int32_t
