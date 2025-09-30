//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: if (0)
    clift.if {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    } {
      // CHECK: 1;
      clift.expr {
        %0 = clift.imm 1 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: else if (2)
    } else {
      clift.if {
        %0 = clift.imm 2 : !int32_t
        clift.yield %0 : !int32_t
      } {
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

    // CHECK: if (5) {
    clift.if {
      %0 = clift.imm 5 : !int32_t
      clift.yield %0 : !int32_t
    } {
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
    // CHECK: } else if (8) {
    } else {
      clift.if {
        %0 = clift.imm 8 : !int32_t
        clift.yield %0 : !int32_t
      } {
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

    // CHECK: if (13) {
    clift.if {
      %0 = clift.imm 13 : !int32_t
      clift.yield %0 : !int32_t
    } {
      // CHECK: if (14)
      clift.if {
        %0 = clift.imm 14 : !int32_t
        clift.yield %0 : !int32_t
      } {
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
