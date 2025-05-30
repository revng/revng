//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!int32_t$ptr)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t$ptr) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: clift.indirection
      %0 = clift.indirection %arg0 : !int32_t$ptr
      // CHECK-NOT: clift.addressof
      %1 = clift.addressof %0 : !int32_t$ptr
      // CHECK: clift.yield %arg0
      clift.yield %1 : !int32_t$ptr
    }
    // CHECK: }
  }
}
