//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!array = !clift.array<1 x !int32_t>

// CHECK: expression must have void or value type
clift.return {
  %0 = clift.undef : !array
  clift.yield %0 : !array
}
