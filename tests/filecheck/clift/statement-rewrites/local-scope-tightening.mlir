//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --tighten-variable-scopes | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f1<!f>() {
    // CHECK: [[L1:%[0-9]+]] = clift.local : !int32_t
    %0 = clift.local : !int32_t
    // CHECK-NOT: clift.local :
    %1 = clift.local : !int32_t
    %2 = clift.local : !int32_t
    %3 = clift.local : !int32_t

    // CHECK: clift.if {
    clift.if {
      %4 = clift.undef : !int32_t
      clift.yield %4 : !int32_t
    // CHECK: } then {
    } then {
      clift.expr {
        clift.yield %0 : !int32_t
      }
    // CHECK: } else {
    } else {
      // CHECK: clift.expr {
      clift.expr {
        clift.yield %0 : !int32_t
      // CHECK: }
      }
      // CHECK: [[L2:%[0-9]+]] = clift.local : !int32_t
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: clift.yield [[L2]] : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.if {
      clift.if {
        %5 = clift.undef : !int32_t
        clift.yield %5 : !int32_t
      // CHECK: } then {
      } then {
        // CHECK: [[L3:%[0-9]+]] = clift.local : !int32_t
        // CHECK: clift.expr {
        clift.expr {
          // CHECK: clift.yield [[L3]] : !int32_t
          clift.yield %2 : !int32_t
        // CHECK: }
        }
      // CHECK: }
      }
    // CHECK: }
    }
    // CHECK: [[L4:%[0-9]+]] = clift.local : !int32_t
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: clift.yield [[L4]] : !int32_t
      clift.yield %3 : !int32_t
    // CHECK: }
    }
  // CHECK: }
  }

  clift.func @f2<!f>() {
    // CHECK: [[L1:%[0-9]+]] = clift.local : !int32_t
    %0 = clift.local : !int32_t
    // CHECK: [[L2:%[0-9]+]] = clift.local : !int32_t
    %1 = clift.local : !int32_t
    // CHECK-NOT: clift.local :
    %2 = clift.local : !int32_t

    // CHECK: clift.if {
    clift.if {
      clift.yield %0 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.expr {
      clift.expr {
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: [[L3:%[0-9]+]] = clift.local : !int32_t
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: clift.yield [[L3]] : !int32_t
        clift.yield %2 : !int32_t
      // CHECK: }
      }
    // CHECK: } else {
    } else {
      clift.expr {
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    // CHECK: }
    }
  // CHECK: }
  }

  clift.func @f3<!f>() {
    // CHECK-NOT: clift.local :
    %0 = clift.local : !int32_t
    // CHECK: clift.expr {
    clift.expr {
      %1 = clift.imm 1 : !int32_t
      clift.yield %1 : !int32_t
    // CHECK: }
    }
    // CHECK: [[L:%[0-9]+]] = clift.local : !int32_t
    // CHECK: clift.if {
    clift.if {
      %2 = clift.imm 1 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.if {
      clift.if {
        %3 = clift.imm 1 : !int32_t
        clift.yield %3 : !int32_t
      // CHECK: } then {
      } then {
        // CHECK: clift.expr {
        clift.expr {
          // CHECK: clift.yield [[L]] : !int32_t
          clift.yield %0 : !int32_t
        // CHECK: }
        }
      // CHECK: }
      }
    // CHECK: }
    }
    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield [[L]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } then {
    } then {
    // CHECK: }
    }
  // CHECK: }
  }

  clift.func @f4<!f>() {
    // CHECK-NOT: clift.local :
    %0 = clift.local : !int32_t

    // CHECK: clift.expr {
    clift.expr {
      %1 = clift.imm 1 : !int32_t
      clift.yield %1 : !int32_t
    // CHECK: }
    }
    // CHECK: [[L:%[0-9]+]] = clift.local : !int32_t
    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield [[L]] : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } then {
    } then {
    // CHECK: }
    }
    // CHECK: clift.if {
    clift.if {
      %2 = clift.imm 1 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.if {
      clift.if {
        %3 = clift.imm 1 : !int32_t
        clift.yield %3 : !int32_t
      // CHECK: } then {
      } then {
        // CHECK: clift.expr {
        clift.expr {
          // CHECK: clift.yield [[L]] : !int32_t
          clift.yield %0 : !int32_t
        // CHECK: }
        }
      // CHECK: }
      }
    // CHECK: }
    }
  // CHECK: }
  }
}
