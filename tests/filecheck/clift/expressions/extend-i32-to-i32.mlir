//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

%value = clift.undef : !int32_t

// CHECK: failed to verify that the sizes of value and result are ordered
clift.extend %value : !int32_t -> !int32_t
