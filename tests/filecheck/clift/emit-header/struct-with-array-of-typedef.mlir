//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s

// The element type of an array has to be complete, so the struct the typedef
// stands for has to be defined before the struct holding the array.

// CHECK: struct _PACKED _SIZE(1) element {
// CHECK: struct _PACKED _SIZE(1) container {

!uint8_t = !clift.int<unsigned 1>
!element = !clift.struct<
  "/type-definition/2-StructDefinition" as "element" : size(1) {
    "/struct-field/2-StructDefinition/0" as "value" : offset(0) !uint8_t
  }
>
!element_alias = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "element_alias" : !element
>
!container = !clift.struct<
  "/type-definition/0-StructDefinition" as "container" : size(1) {
    "/struct-field/0-StructDefinition/0" as "items" : offset(0) !clift.array<1 x !element_alias>
  }
>
module attributes {clift.module, clift.types = [!container]} {
}
