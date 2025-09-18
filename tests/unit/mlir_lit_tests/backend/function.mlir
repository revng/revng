//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "fun_0x40001001_t" : !void()
>

!f$ptr = !clift.ptr<8 to !f>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: fun_0x40001001_t *var_0 = fun_0x40001001;
    clift.local : !f$ptr = {
      %f = clift.use @fun_0x40001001 : !f
      %r = clift.cast<decay> %f : !f -> !f$ptr
      clift.yield %r : !f$ptr
    } attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      clift.name = "var_0"
    }
  }
  // CHECK: }
}
