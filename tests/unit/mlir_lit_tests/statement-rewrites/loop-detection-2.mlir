//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --loop-detection --beautify-statements | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" : !void(!int32_t, !int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: [[L:%[0-9]+]] = clift.make_label
    %label_0 = clift.make_label

    // CHECK-NOT: clift.make_label
    %label_1 = clift.make_label

    clift.assign_label %label_0

    // CHECK-NEXT: clift.do_while {

    // CHECK-NEXT: clift.expr {
    clift.expr {
      // CHECK: clift.imm 0
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK-NEXT: clift.if
    clift.if {
      // CHECK: clift.yield %arg0
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK-NEXT: clift.goto [[L]]
      // CHECK-NEXT: }
    } else {
      // CHECK-NEXT: clift.expr {
      clift.expr {
        // CHECK: clift.imm 1
        %0 = clift.imm 1 : !int32_t
        clift.yield %0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.if
      clift.if {
        clift.yield %arg1 : !int32_t
      } {
      } else {
        clift.goto %label_0
      }

      // CHECK-NEXT: } {
      // WIP: Match !arg1, with any odd number of negations
      // CHECK: }

      // CHECK-NEXT: clift.expr {
      clift.expr {
        // CHECK: clift.imm 2
        %0 = clift.imm 2 : !int32_t
        clift.yield %0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.goto
      clift.goto %label_1
    }

    // CHECK-NEXT: clift.assign_label [[L]]
    clift.assign_label %label_1
  }
// CHECK-NEXT: }
}
