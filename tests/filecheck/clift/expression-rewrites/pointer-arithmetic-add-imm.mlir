//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!uintptr_t = !clift.int<unsigned 8>

!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!uintptr_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uintptr_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-DAG: [[A:%[0-9]+]] = clift.imm 5
      %0 = clift.imm 20 : !uintptr_t
      // CHECK-DAG: [[B:%[0-9]+]] = clift.bitcast %arg0
      // CHECK: [[C:%[0-9]+]] = clift.ptr_add [[B]], [[A]]
      %1 = clift.add %arg0, %0 : !uintptr_t
      %2 = clift.bitcast %1 : !uintptr_t -> !int32_t$ptr
      // CHECK: clift.yield [[C]]
      clift.yield %2 : !int32_t$ptr
    }
    // CHECK: }
  }
}
