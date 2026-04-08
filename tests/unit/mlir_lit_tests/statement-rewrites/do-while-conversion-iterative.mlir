//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-statements | FileCheck %s

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
    %continue = clift.make_label
    // CHECK: clift.do_while
    // CHECK-NOT: break
    // CHECK-NOT: continue
    // CHECK-SAME: body {
    clift.while break %break continue %continue cond {
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
        clift.continue_to %continue
      }
      clift.break_to %break

    // CHECK-NEXT: } cond {
      // CHECK: [[COND1:%[0-9]+]] = clift.not %arg0 : !int32_t -> !int8_t
      // CHECK: [[COND2:%[0-9]+]] = clift.not [[COND1]] : !int8_t -> !int8_t
      // CHECK: clift.yield [[COND2]]
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
