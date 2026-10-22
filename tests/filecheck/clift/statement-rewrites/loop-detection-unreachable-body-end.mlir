//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --loop-detection --canonicalize | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) {
    %loop = clift.make_label
    %elsewhere = clift.make_label

    clift.assign_label %loop
    // COM: neither branch of the if-statement below falls through, so the end
    // COM: of the loop body cannot be reached and no break closes it
    // CHECK: clift.while
    // CHECK-NOT: break
    // CHECK: continue [[CONTINUE:%[0-9]+]]
    // CHECK-SAME: cond {
      // CHECK: [[COND:%[0-9]+]] = clift.true
      // CHECK: clift.yield [[COND]] : !clift.bool
    // CHECK: } body {

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[IMM:%[0-9]+]] = clift.imm 10 : !int32_t
      %0 = clift.imm 10 : !int32_t
      // CHECK: clift.yield [[IMM]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      // CHECK: [[IF_COND:%[0-9]+]] = clift.test %arg0 : !int32_t
      %0 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield [[IF_COND]] : !clift.bool
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.continue_to [[CONTINUE]]
      clift.goto %loop
    // CHECK: } else {
    } else {
      // CHECK: clift.goto [[ELSEWHERE:%[0-9]+]]
      clift.goto %elsewhere
    // CHECK: }
    }
    // CHECK-NOT: clift.break_to
    // COM: loop closer
    // CHECK: }

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[IMM:%[0-9]+]] = clift.imm 20 : !int32_t
      %0 = clift.imm 20 : !int32_t
      // CHECK: clift.yield [[IMM]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK: clift.assign_label [[ELSEWHERE]]
    clift.assign_label %elsewhere
    // CHECK: clift.return
    clift.return {}
  // CHECK: }
  }
// CHECK: }
}
