//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!uintptr_t = !clift.primitive<unsigned 8>

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!uintptr_t, !uintptr_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uintptr_t, %arg1 : !uintptr_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-DAG: [[A:%[0-9]+]] = clift.imm 5
      %0 = clift.imm 20 : !uintptr_t
      // CHECK-DAG: [[B:%[0-9]+]] = clift.mul %arg1, [[A]]
      %1 = clift.mul %arg1, %0 : !uintptr_t
      // CHECK-DAG: [[C:%[0-9]+]] = clift.cast<bitcast> %arg0
      // CHECK: [[D:%[0-9]+]] = clift.ptr_add [[C]], [[B]]
      %2 = clift.add %arg0, %1 : !uintptr_t
      %3 = clift.cast<bitcast> %2 : !uintptr_t -> !int32_t$ptr
      // CHECK: clift.yield [[D]]
      clift.yield %3 : !int32_t$ptr
    }
    // CHECK: }
  }
}
