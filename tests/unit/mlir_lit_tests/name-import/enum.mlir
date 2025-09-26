//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-model-names %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.primitive<void 0>
!uint8_t = !clift.primitive<unsigned 1>

!g = !clift.func<"/type-definition/1003-CABIFunctionDefinition" : !void()>

// CHECK: !e = !clift.enum<"/type-definition/2003-EnumDefinition" as "e" : !uint8_t {
// CHECK:   "/enum-entry/2003-EnumDefinition/0" as "e_0" : 0
// CHECK: }>
!e = !clift.enum<"/type-definition/2003-EnumDefinition" : !uint8_t {
  "/enum-entry/2003-EnumDefinition/0" : 0
}>

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001003<!g>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001003:Code_x86_64"
  // CHECK: }
  clift.func @g<!g>() attributes {
    handle = "/function/0x40001003:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !e
    %0 = clift.local : !e
  }
}
