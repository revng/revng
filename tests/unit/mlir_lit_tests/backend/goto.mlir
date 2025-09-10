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
    %exit = clift.make_label {
      clift.handle = "/goto-label/0x40001001:Code_x86_64/0",
      clift.name = "label_0"
    }
    // CHECK: 0;
    clift.expr {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
    }
    // CHECK: goto label_0;
    clift.goto %exit
    // CHECK: label_0: ;
    clift.assign_label %exit
  }
  // CHECK: }
}
