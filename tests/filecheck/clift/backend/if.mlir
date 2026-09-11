//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: if (true)
    clift.if {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } then {
      // CHECK: 1;
      clift.expr {
        %0 = clift.imm 1 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: else if (true)
    } else {
      clift.if {
        %0 = clift.true
        clift.yield %0 : !clift.bool
      } then {
        // CHECK: 3;
        clift.expr {
          %0 = clift.imm 3 : !int32_t
          clift.yield %0 : !int32_t
        }
      // CHECK: else
      } else {
        // CHECK: 4;
        clift.expr {
          %0 = clift.imm 4 : !int32_t
          clift.yield %0 : !int32_t
        }
      }
    }

    // CHECK: if (true) {
    clift.if {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } then {
      // CHECK: 6;
      clift.expr {
        %0 = clift.imm 6 : !int32_t
        clift.yield %0 : !int32_t
      }
      // CHECK: 7;
      clift.expr {
        %0 = clift.imm 7 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: } else if (true) {
    } else {
      clift.if {
        %0 = clift.true
        clift.yield %0 : !clift.bool
      } then {
        // CHECK: 9;
        clift.expr {
          %0 = clift.imm 9 : !int32_t
          clift.yield %0 : !int32_t
        }
        // CHECK: 10;
        clift.expr {
          %0 = clift.imm 10 : !int32_t
          clift.yield %0 : !int32_t
        }
      } else {
        // CHECK: 11;
        clift.expr {
          %0 = clift.imm 11 : !int32_t
          clift.yield %0 : !int32_t
        }
        // CHECK: 12;
        clift.expr {
          %0 = clift.imm 12 : !int32_t
          clift.yield %0 : !int32_t
        }
      }
    }
    // CHECK: }

    // CHECK: if (true) {
    clift.if {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } then {
      // CHECK: if (true)
      clift.if {
        %0 = clift.true
        clift.yield %0 : !clift.bool
      } then {
        // CHECK: 15;
        clift.expr {
          %0 = clift.imm 15 : !int32_t
          clift.yield %0 : !int32_t
        }
      }
    // CHECK: } else
    } else {
      // CHECK: 16;
      clift.expr {
        %0 = clift.imm 16 : !int32_t
        clift.yield %0 : !int32_t
      }
    }
  }
  // CHECK: }
}
