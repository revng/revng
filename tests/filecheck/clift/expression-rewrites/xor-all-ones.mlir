//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<"/model-type/1001" : !void(!uint32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uint32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.imm 0xFFFFFFFF : !uint32_t
      // CHECK: [[X:%[0-9]+]] = clift.bitnot %arg0 : !uint32_t
      %1 = clift.bitxor %arg0, %0 : !uint32_t
      // CHECK: clift.yield [[X]]
      clift.yield %1 : !uint32_t
    }
    // CHECK: }
  }
}
