//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

%value = clift.undef : !float32_t

// CHECK: operand #0 must be integer type
clift.stof %value : !float32_t -> !float64_t
