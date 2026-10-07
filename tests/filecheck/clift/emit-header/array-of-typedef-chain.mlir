//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s

// The complete type has to be found across a whole chain of typedefs, not just
// the first one.

// CHECK: struct _PACKED _SIZE(1) element {
// CHECK: struct _PACKED _SIZE(1) container {

!uint8_t = !clift.int<unsigned 1>
!element = !clift.struct<
  "/type-definition/3-StructDefinition" as "element" : size(1) {
    "/struct-field/3-StructDefinition/0" as "value" : offset(0) !uint8_t
  }
>
!first = !clift.typedef<
  "/type-definition/2-TypedefDefinition" as "first" : !element
>
!second = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "second" : !first
>
!container = !clift.struct<
  "/type-definition/0-StructDefinition" as "container" : size(1) {
    "/struct-field/0-StructDefinition/0" as "items" : offset(0) !clift.array<1 x !second>
  }
>
module attributes {clift.module, clift.types = [!container]} {
}
