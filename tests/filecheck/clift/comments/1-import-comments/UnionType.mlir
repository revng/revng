//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng pipeline run-pipe import-descriptive-info %S/../0-import-types/UnionType.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: !my_union = !clift.union<
// CHECK:   "/type-definition/0-UnionDefinition" as "my_union" : {
// CHECK:     "/union-field/0-UnionDefinition/0" as "member_0" : !uint8_t
// CHECK:       comment "one-byte field",
// CHECK:     "/union-field/0-UnionDefinition/1" as "member_1" : !uint16_t
// CHECK:       comment "two-byte field",
// CHECK:     "/union-field/0-UnionDefinition/2" as "member_2" : !uint32_t
// CHECK:       comment "four-byte field\0A\0ATime to go looking for the three-byte one! /j"
// CHECK:   }
// CHECK:   comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure union-attached comments don't\0Aaccidentally get broken!"
// CHECK: >

!_type_definition_0_UnionDefinition = !clift.union<
  "/type-definition/0-UnionDefinition" : {
    "/union-field/0-UnionDefinition/0" : !uint8_t,
    "/union-field/0-UnionDefinition/1" : !uint16_t,
    "/union-field/0-UnionDefinition/2" : !uint32_t
  }
>

module attributes {clift.module, clift.types = [!_type_definition_0_UnionDefinition]} {
}
