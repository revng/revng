//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.if {
    clift.if {
      // CHECK: %0 = clift.not %arg0 : !int32_t -> !int8_t
      // CHECK: clift.yield %0 : !int8_t
      clift.yield %arg0 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.expr {
        // CHECK: %0 = clift.imm 10 : !int32_t
        // CHECK: clift.yield %0 : !int32_t
      // CHECK: }
    // CHECK-NOT: } else {
    } else {
      clift.expr {
        %0 = clift.imm 10 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
