//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-model-names %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.primitive<void 0>
!uint8_t = !clift.primitive<unsigned 1>

// CHECK: !f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" as "f" :
// CHECK:   !void(!uint8_t)
// CHECK: >
!f = !clift.func<"/type-definition/1001-CABIFunctionDefinition" : !void(!uint8_t)>

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
    // CHECK: clift.make_label {
    // CHECK:   clift.handle = "/goto-label/0x40001001:Code_x86_64/0"
    // CHECK:   clift.name = "label_0"
    // CHECK: }
    %l = clift.make_label

    // CHECK: clift.local : !uint8_t attributes {
    // CHECK:   clift.handle = "/local-variable/0x40001001:Code_x86_64/0"
    // CHECK:   clift.name = "var_0"
    // CHECK: }
    %1 = clift.local : !uint8_t
  }
}
