//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!float32_t = !clift.float<4>

%value = clift.undef : !float32_t

// CHECK: result #0 must be non-const floating point type
clift.ftof %value : !float32_t -> !int32_t
