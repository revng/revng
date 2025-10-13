//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %continue = clift.make_label {
      clift.handle = "/goto-label/0x40001001:Code_x86_64/0",
      clift.name = "continue_label"
    }

    // CHECK: for (;;) {
    clift.for continue %continue body {
      // CHECK: continue_to continue_label;
      clift.continue_to %continue
      // CHECK: continue_label: ;
    }
    // CHECK: }
  }
  // CHECK: }
}
