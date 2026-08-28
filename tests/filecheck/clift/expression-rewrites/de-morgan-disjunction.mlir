//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t, !int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.test %arg0 : !int32_t
      %0 = clift.test %arg0 : !int32_t
      %1 = clift.not %0
      // CHECK: %1 = clift.test %arg1 : !int32_t
      %2 = clift.test %arg1 : !int32_t
      %3 = clift.not %2
      // CHECK: %2 = clift.and %0, %1
      %4 = clift.or %1, %3
      %5 = clift.not %4
      // CHECK: clift.yield %2 : !clift.bool
      clift.yield %5 : !clift.bool
    // CHECK: }
    }
  }
}
