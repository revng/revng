//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>

%i = clift.undef : !int16_t

// CHECK: failed to verify that all of {value, result} have same object size
clift.bitcast %i : !int16_t -> !int32_t
