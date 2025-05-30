//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: clift.neg
      %0 = clift.neg %arg0 : !int32_t
      %1 = clift.neg %0 : !int32_t
      // CHECK: clift.yield %arg0
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}
