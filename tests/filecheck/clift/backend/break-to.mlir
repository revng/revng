//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %break = clift.make_label {
      handle = "/goto-label/0x40001001:Code_x86_64/0",
      name = "break_label"
    }

    // CHECK: for (;;)
    clift.for break %break body {
      // CHECK: break_to break_label;
      clift.break_to %break
    }
    // CHECK: break_label: ;
  }
  // CHECK: }
}
