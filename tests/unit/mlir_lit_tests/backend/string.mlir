//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>
!char$const = !clift.const<!clift.primitive<number 1>>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: "hello";
    clift.expr {
      %s = clift.str "hello" : !clift.array<6 x !char$const>
      clift.yield %s : !clift.array<6 x !char$const>
    }
  }
  // CHECK: }
}
