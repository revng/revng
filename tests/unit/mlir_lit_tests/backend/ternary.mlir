//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

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
    // CHECK: 0 ? 1 : 2;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %r = clift.ternary %0, %1, %2 : (!int32_t, !int32_t)
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
