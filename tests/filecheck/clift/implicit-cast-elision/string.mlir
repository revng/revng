//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --elide-implicit-casts | FileCheck %s

!int8_t = !clift.int<signed 1>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !clift.ptr<8 to !clift.const<!int8_t>>()
>

#c_dialect = #clift.c_dialect<
  ImplicitPointerSignConversions = true
>

module attributes {clift.module, clift.c_dialect = #c_dialect} {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64",
    clift.legalized
  } {
    // CHECK: clift.return {
    clift.return {
      // CHECK: %0 = clift.str "hello" : !clift.const<!clift.array<6 x !clift.c_char<1>>>
      %0 = clift.str "hello" : !clift.const<!clift.array<6 x !clift.c_char<1>>>
      // CHECK: %1 = clift.implicit_cast %0 : !clift.const<!clift.array<6 x !clift.c_char<1>>> -> !clift.ptr<8 to !clift.const<!int8_t>>
      %1 = clift.decay %0 : !clift.const<!clift.array<6 x !clift.c_char<1>>> -> !clift.ptr<8 to !clift.const<!int8_t>>
      // CHECK: clift.yield %1 : !clift.ptr<8 to !clift.const<!int8_t>>
      clift.yield %1 : !clift.ptr<8 to !clift.const<!int8_t>>
    // CHECK: }
    }
  }
}
