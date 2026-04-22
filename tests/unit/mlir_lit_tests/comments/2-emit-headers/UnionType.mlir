//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-type-and-global-header=ptml %s -o /dev/null | %revngptml | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: /// Take a look at struct and function comment tests for more "meat".
// CHECK: ///
// CHECK: /// This one is just to ensure union-attached comments don't
// CHECK: /// accidentally get broken!
// CHECK: union _PACKED my_union {
//
// CHECK:   /// one-byte field
// CHECK:   uint8_t member_0;
//
// CHECK:   /// two-byte field
// CHECK:   uint16_t member_1;
//
// CHECK:   /// four-byte field
// CHECK:   ///
// CHECK:   /// Time to go looking for the three-byte one! /j
// CHECK:   uint32_t member_2;
// CHECK: };

!my_union = !clift.union<
  "/type-definition/0-UnionDefinition" as "my_union" : {
    "/union-field/0-UnionDefinition/0" as "member_0" : !uint8_t
      comment "one-byte field",
    "/union-field/0-UnionDefinition/1" as "member_1" : !uint16_t
      comment "two-byte field",
    "/union-field/0-UnionDefinition/2" as "member_2" : !uint32_t
      comment "four-byte field\0A\0ATime to go looking for the three-byte one! /j"
  }
  comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure union-attached comments don't\0Aaccidentally get broken!"
>

module attributes {clift.module, clift.types = [!my_union]} {
}
