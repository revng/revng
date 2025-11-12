//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --loop-detection --canonicalize | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" : !void(!int32_t, !int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) {
    %outer_loop = clift.make_label
    %inner_loop = clift.make_label

    clift.assign_label %outer_loop
    // CHECK: clift.while
    // CHECK: break [[OUTER_BREAK:%[0-9]+]]
    // CHECK: continue [[OUTER_CONTINUE:%[0-9]+]]
    // CHECK-SAME: cond {
      // CHECK: [[COND:%[0-9]+]] = clift.imm 1 : !int8_t
      // CHECK: clift.yield [[COND]] : !int8_t
    // CHECK: } body {

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[IMM:%[0-9]+]] = clift.imm 10 : !int32_t
      %0 = clift.imm 10 : !int32_t
      // CHECK: clift.yield [[IMM]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    clift.assign_label %inner_loop
    // CHECK: clift.while
    // CHECK: break [[INNER_BREAK:%[0-9]+]]
    // CHECK: continue [[INNER_CONTINUE:%[0-9]+]]
    // CHECK: cond {
      // CHECK: [[COND:%[0-9]+]] = clift.imm 1 : !int8_t
      // CHECK: clift.yield [[COND]] : !int8_t
    // CHECK: } body {

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[IMM:%[0-9]+]] = clift.imm 20 : !int32_t
      %0 = clift.imm 20 : !int32_t
      // CHECK: clift.yield [[IMM]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.continue_to [[INNER_CONTINUE]]
      clift.goto %inner_loop
    // CHECK: }
    }

    // CHECK: clift.break_to [[INNER_BREAK]]
    // COM: inner loop closer
    // CHECK: }

    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield %arg1 : !int32_t
      clift.yield %arg1 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.continue_to [[OUTER_CONTINUE]]
      clift.goto %outer_loop
    // CHECK: }
    }

    // CHECK: clift.break_to [[OUTER_BREAK]]
    // COM: outer loop closer
    // CHECK: }
  // CHECK: }
  }
// CHECK-NEXT: }
}
