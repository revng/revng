//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-do-while-conditions | FileCheck %s

!void = !clift.void
!int8_t = !clift.int<signed 1>
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.make_label
    %break = clift.make_label
    // CHECK: clift.do_while break %0 body {
    clift.while break %break cond {
      %0 = clift.imm 1 : !int32_t
      clift.yield %0 : !int32_t
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
        clift.yield %arg0 : !int32_t
      } then {
        clift.break_to %break
      }

    // CHECK-NEXT: } cond {
      // CHECK: [[COND:%[0-9]+]] = clift.not %arg0 : !int32_t -> !int8_t
      // CHECK: clift.yield [[COND]]
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
