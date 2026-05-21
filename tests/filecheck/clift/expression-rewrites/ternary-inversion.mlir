//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 0 : !int32_t
      %0 = clift.imm 0 : !int32_t
      // CHECK-NOT: clift.not
      %1 = clift.not %0 : !int32_t -> !int32_t
      // CHECK: %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 1 : !int32_t
      // CHECK: %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 2 : !int32_t
      // CHECK: %3 = clift.ternary %0, %2, %1 : (!int32_t,!int32_t)
      %4 = clift.ternary %1, %2, %3 : (!int32_t, !int32_t)
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %4 : !int32_t
    }
    // CHECK: }
  }
}
