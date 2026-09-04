//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "fun_0x40001001_t" : !void()
>

!f$ptr = !clift.ptr<8 to !f>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: fun_0x40001001_t *var_0 = (fun_0x40001001_t *) fun_0x40001001;
    clift.local : !f$ptr = {
      %f = clift.use @fun_0x40001001 : !f
      %r = clift.decay %f : !f -> !f$ptr
      clift.yield %r : !f$ptr
    } attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }

    // CHECK: fun_0x40001001_t *var_1 = fun_0x40001001;
    clift.local : !f$ptr = {
      %f = clift.use @fun_0x40001001 : !f
      %r = clift.implicit_cast %f : !f -> !f$ptr
      clift.yield %r : !f$ptr
    } attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      name = "var_1"
    }
  }
  // CHECK: }
}
