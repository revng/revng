//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s 2>&1 --emit-bytecode | %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void

// CHECK: !_type_definition_1_StructDefinition = !clift.struct
// CHECK: <
// CHECK:   "/type-definition/1-StructDefinition" : size(8)
// CHECK:   {
// CHECK:     "/struct-field/1-StructDefinition/0" : offset(0)
// CHECK:     !clift.ptr<8 to !clift.const<!void>>
// CHECK:   }
// CHECK: >

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(8) {
    "/struct-field/1-StructDefinition/0" : offset(0) !clift.ptr<8 to !clift.const<!void>>
  }
>

module attributes {clift.module, clift.test = !s} {}
