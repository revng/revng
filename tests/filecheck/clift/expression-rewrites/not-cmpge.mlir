//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.slt %arg0, %arg0 : !int32_t
      %0 = clift.sge %arg0, %arg0 : !int32_t
      // CHECK-NOT: clift.not
      %1 = clift.not %0
      // CHECK: clift.yield %0 : !clift.bool
      clift.yield %1 : !clift.bool
    }
    // CHECK: }
  }
}
