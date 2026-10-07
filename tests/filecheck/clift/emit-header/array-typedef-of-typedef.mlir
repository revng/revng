//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s

// An array typedef needs a complete element type to be declared at all, so the
// struct has to come before the typedef, even though a typedef never gets a
// separate forward declaration.

// CHECK: struct _PACKED _SIZE(1) element {
// CHECK: typedef element_alias array_alias[1];

!uint8_t = !clift.int<unsigned 1>
!element = !clift.struct<
  "/type-definition/2-StructDefinition" as "element" : size(1) {
    "/struct-field/2-StructDefinition/0" as "value" : offset(0) !uint8_t
  }
>
!element_alias = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "element_alias" : !element
>
!array_alias = !clift.typedef<
  "/type-definition/0-TypedefDefinition" as "array_alias" : !clift.array<1 x !element_alias>
>
module attributes {clift.module, clift.types = [!array_alias]} {
}
