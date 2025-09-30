//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

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
    // CHECK: "hello \"world\" \\ \0\x07\x08\t\n\v\f\r\x7f\xff";
    clift.expr {
      %s = clift.str "hello \"world\" \\ \00\07\08\09\0a\0b\0c\0d\7f\ff" : !clift.array<27 x !char$const>
      clift.yield %s : !clift.array<27 x !char$const>
    }
  }
  // CHECK: }
}
