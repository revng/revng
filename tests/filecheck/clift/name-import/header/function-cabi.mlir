//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// TODO: this test should start *only* from the model, as the clift for it can
//       easily be generated from it.

// RUN: %root/bin/revng pipe import-descriptive-info %S/../model.yml %s /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>

// CHECK: !f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" as "f" :
// CHECK:   !void(!uint8_t)
// CHECK:   [
// CHECK:     #clift.c_attribute<"_ABI" : "/macro/_ABI"
// CHECK:     [
// CHECK:       #clift.identifier<"SystemV_x86_64">
// CHECK:     ]>
// CHECK:   ]
// CHECK: >
!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition"
  : !void(!uint8_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>
  ]
>

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001001<!f>(
  // CHECK:   %arg0: !uint8_t {
  // CHECK:     clift.handle = "/cabi-argument/1001-CABIFunctionDefinition/0"
  // CHECK:     clift.name = "a"
  // CHECK:   }
  // CHECK: ) -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001001:Code_x86_64"
  // CHECK: }
  clift.func @f<!f>(%arg0 : !uint8_t { clift.handle = "/cabi-argument/1001-CABIFunctionDefinition/0" }) attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
  }
}
