//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %S/emit-strings %root/bin/revng %S/model.yml %s /function/0x401000:Code_x86_64 | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint32_t = !clift.int<unsigned 4>

!seg_0x402000 = !clift.struct<
  "/type-definition/2000-StructDefinition" : size(16) {
    "/struct-field/2000-StructDefinition/0" : offset(0) !uint32_t,
    "/struct-field/2000-StructDefinition/4" : offset(4) !clift.array<12 x !uint8_t>
  }
>

!fun_0x401000 = !clift.func<"/type-definition/1000-CABIFunctionDefinition" : !clift.void()>

module attributes {clift.module} {
  clift.global @seg_0x402000 : !clift.const<!seg_0x402000> attributes {
    handle = "/segment/0x402000:Generic64"
  }

  clift.func @fun_0x401000<!fun_0x401000>() attributes {
    handle = "/function/0x401000:Code_x86_64"
  } {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.str "hello world" : !clift.const<!clift.array<12 x !uint8_t>>
      %0 = clift.use @seg_0x402000 : !clift.const<!seg_0x402000>
      %1 = clift.access<1> %0 : !clift.const<!seg_0x402000> -> !clift.const<!clift.array<12 x !uint8_t>>
      // CHECK: clift.yield %0 : !clift.const<!clift.array<12 x !uint8_t>>
      clift.yield %1 : !clift.const<!clift.array<12 x !uint8_t>>
    // CHECK: }
    }
  }
}
