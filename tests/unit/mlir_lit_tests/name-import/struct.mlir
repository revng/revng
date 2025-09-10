//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-model-names %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.primitive<void 0>
!uint8_t = !clift.primitive<unsigned 1>

!g = !clift.func<"/type-definition/1003-CABIFunctionDefinition" : !void()>

// CHECK: !s = !clift.struct<"/type-definition/2001-StructDefinition" as "s" : size(1) {
// CHECK:   "/struct-field/2001-StructDefinition/0" as "x" : offset(0) !uint8_t
// CHECK: }>
!s = !clift.struct<"/type-definition/2001-StructDefinition" : size(1) {
  "/struct-field/2001-StructDefinition/0" : offset(0) !uint8_t
}>

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001003<!g>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001003:Code_x86_64"
  // CHECK: }
  clift.func @g<!g>() attributes {
    handle = "/function/0x40001003:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !s
    %0 = clift.local : !s
  }
}
