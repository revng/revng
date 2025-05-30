//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-statements | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!int32_t$ptr)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t$ptr) -> !void {
    // CHECK: clift.if {
    clift.if {
      %0 = clift.imm 1 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.expr {
      clift.expr {
        %0 = clift.indirection %arg0 : !int32_t$ptr
        %1 = clift.imm 0 : !int32_t
        %2 = clift.assign %0, %1 : !int32_t
        clift.yield %2 : !int32_t
      // CHECK: }
      }
    // CHECK: }
    // CHECK-NOT: else
    } else {
    ^bb0:
    }
  // CHECK: }
  }
// CHECK: }
}
