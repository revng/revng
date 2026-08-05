//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// TODO: this test should start *only* from the model, as the clift for it can
//       easily be generated from it.

// RUN: %root/bin/revng pipeline run-pipe import-descriptive-info %S/../model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>

!g = !clift.func<"/type-definition/1003-CABIFunctionDefinition" : !void()>

// CHECK: !u = !clift.union<"/type-definition/2002-UnionDefinition" as "u" : {
// CHECK:   "/union-field/2002-UnionDefinition/0" as "x" : !uint8_t
// CHECK: }>
!u = !clift.union<"/type-definition/2002-UnionDefinition" : {
  "/union-field/2002-UnionDefinition/0" : !uint8_t
}>

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001003<!g>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001003:Code_x86_64"
  // CHECK: }
  clift.func @g<!g>() attributes {
    handle = "/function/0x40001003:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !u
    %0 = clift.local : !u
  }
}
