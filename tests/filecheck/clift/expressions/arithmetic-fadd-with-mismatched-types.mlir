//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

%f = clift.undef : !float32_t
%d = clift.undef : !float64_t

// CHECK: failed to verify that all of {lhs, rhs, result} have same type
clift.fadd %f, %d : (!float32_t, !float64_t) -> !float32_t
