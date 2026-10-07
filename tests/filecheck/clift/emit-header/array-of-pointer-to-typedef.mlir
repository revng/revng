//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s

// A pointer does not need a complete pointee, so an array of pointers to a
// typedef must not force the definition of what the typedef stands for: the
// forward declaration is enough and the two can be defined in any order.

// CHECK-DAG: struct _PACKED _SIZE(8) container {
// CHECK-DAG: struct _PACKED _SIZE(1) element {

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
  "/type-definition/0-StructDefinition" as "container" : size(8) {
    "/struct-field/0-StructDefinition/0" as "items" : offset(0) !clift.array<1 x !clift.ptr<8 to !element_alias>>
  }
>
module attributes {clift.module, clift.types = [!container]} {
}
