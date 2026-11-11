//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!float32_t = !clift.float<4>

%value = clift.undef : !float32_t

// CHECK: failed to verify that the sizes of value and result are not equal
clift.ftof %value : !float32_t -> !float32_t
