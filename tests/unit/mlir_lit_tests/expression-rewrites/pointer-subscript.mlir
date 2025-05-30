//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!uintptr_t = !clift.primitive<unsigned 8>

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!int32_t$ptr, !uintptr_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t$ptr, %arg1 : !uintptr_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.subscript %arg0, %arg1
      %0 = clift.ptr_add %arg0, %arg1 : (!int32_t$ptr, !uintptr_t)
      // CHECK-NOT: clift.indirection
      %1 = clift.indirection %0 : !int32_t$ptr
      // CHECK: clift.yield %0
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}
