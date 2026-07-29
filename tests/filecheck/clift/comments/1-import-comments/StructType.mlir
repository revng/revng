//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe import-descriptive-info %S/../0-import-types/StructType.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: !my_commented_struct = !clift.struct<
// CHECK:   "/type-definition/0-StructDefinition" as "my_commented_struct" : size(8) {
// CHECK:     "/struct-field/0-StructDefinition/0" as "offset_0" : offset(0) !uint8_t
// CHECK:       comment "This is the very first field, the comment for it is pretty short.",
// CHECK:     "/struct-field/0-StructDefinition/2" as "offset_2" : offset(2) !uint16_t
// CHECK:       comment "This second field is extremely important, so the user put a whole lot of effort into this comment. It ended up really long. But it also turned out messy and mis-formatted! This line is so long, it's crazy! I wonder if we should help the user by introducing the extra formatting this so desperately needs!",
// CHECK:     "/struct-field/0-StructDefinition/4" as "offset_4" : offset(4) !uint32_t
// CHECK:       comment "Unlike the previous one, this comment is short, but it contains a code\0Asnippet! So we should be careful and not mess this one up!\0A```cpp\0A  int *a = nullopt;\0A  *a = 42;\0A```\0A\0A(Your code can have no UB if you write nothing but comments!)\0A\0AP. S. This field is also twice the size of the other two!"
// CHECK:   }
// CHECK:   comment "This struct has a lot of thoughtful and well argumented comments attached\0Ato it by the user.\0A\0AIt's extremely important we preserve these well and display them nicely.\0A\0AThis comment is also already well-formatted, so there's no extra processing\0Ato be done on it."
// CHECK: >

!_type_definition_0_StructDefinition = !clift.struct<
  "/type-definition/0-StructDefinition" : size(8) {
    "/struct-field/0-StructDefinition/0" : offset(0) !uint8_t,
    "/struct-field/0-StructDefinition/2" : offset(2) !uint16_t,
    "/struct-field/0-StructDefinition/4" : offset(4) !uint32_t
  }
>

module attributes {clift.module, clift.types = [!_type_definition_0_StructDefinition]} {
}
