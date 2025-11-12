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
    %first_loop = clift.make_label
    %second_loop = clift.make_label

    clift.assign_label %first_loop
    // CHECK: clift.while
    // CHECK: break [[FIRST_BREAK:%[0-9]+]]
    // CHECK: continue [[FIRST_CONTINUE:%[0-9]+]]
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

    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield %arg1 : !int32_t
      clift.yield %arg1 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.continue_to [[FIRST_CONTINUE]]
      clift.goto %first_loop
    // CHECK: }
    }

    // CHECK: clift.break_to [[FIRST_BREAK]]
    // COM: first loop closer
    // CHECK: }

    clift.assign_label %second_loop
    // CHECK: clift.while
    // CHECK: break [[SECOND_BREAK:%[0-9]+]]
    // CHECK: continue [[SECOND_CONTINUE:%[0-9]+]]
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
      // CHECK: clift.continue_to [[SECOND_CONTINUE]]
      clift.goto %second_loop
    // CHECK: }
    }

    // CHECK: clift.break_to [[SECOND_BREAK]]
    // COM: second loop closer
    // CHECK: }
  // CHECK: }
  }
// CHECK-NEXT: }
}
