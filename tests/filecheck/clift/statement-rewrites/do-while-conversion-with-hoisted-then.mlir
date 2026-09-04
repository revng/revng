//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-statements=enable-patterns=do-while-conversion --canonicalize | FileCheck %s

!void = !clift.void
!int8_t = !clift.int<signed 1>
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK-NOT: clift.make_label
    %break = clift.make_label
    // CHECK: clift.do_while
    // CHECK-SAME: body {
    clift.while break %break cond {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } body {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: [[A:%[0-9]+]] = clift.imm 10 : !int32_t
        %0 = clift.imm 10 : !int32_t
        // CHECK: clift.yield [[A]] : !int32_t
        clift.yield %0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.if
      clift.if {
        %0 = clift.test %arg0 : !int32_t
        clift.yield %0 : !clift.bool
      } then {
      } else {
        clift.break_to %break
      }

    // CHECK-NEXT: } cond {
      // CHECK: [[COND:%[0-9]+]] = clift.test %arg0 : !int32_t
      // CHECK: clift.yield [[COND]] : !clift.bool
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
