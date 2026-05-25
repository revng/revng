//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>

!s = !clift.struct<
  "/type-definition/2002-StructDefinition" as "s" : size(8) {
    "/struct-field/2002-StructDefinition/0" as "x" : offset(0) !clift.array<2 x !int32_t>
  }
>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: s var_0;
    %var0 = clift.local : !s attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }

    // CHECK: var_0.x[1];
    clift.expr {
      %0 = clift.access<0> %var0 : !s -> !clift.array<2 x !int32_t>
      %1 = clift.decay %0 : !clift.array<2 x !int32_t> -> !clift.ptr<8 to !int32_t>
      %2 = clift.imm 1 : !int32_t
      %3 = clift.subscript %1, %2 : (!clift.ptr<8 to !int32_t>, !int32_t)
      clift.yield %3 : !int32_t
    }
  }
  // CHECK: }
}
