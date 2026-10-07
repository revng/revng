//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s

// The element type of an array typedef is itself an array typedef, which has no
// separate forward declaration either: the chain of definitions has to come out
// innermost first, all the way down to the struct.

// CHECK: struct _PACKED _SIZE(1) element {
// CHECK: typedef element inner[2];
// CHECK: typedef inner outer[3];

!uint8_t = !clift.int<unsigned 1>
!element = !clift.struct<
  "/type-definition/2-StructDefinition" as "element" : size(1) {
    "/struct-field/2-StructDefinition/0" as "value" : offset(0) !uint8_t
  }
>
!inner = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "inner" : !clift.array<2 x !element>
>
!outer = !clift.typedef<
  "/type-definition/0-TypedefDefinition" as "outer" : !clift.array<3 x !inner>
>
module attributes {clift.module, clift.types = [!outer]} {
}
