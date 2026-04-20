//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-descriptive-info %S/../0-import-types/EnumType.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!uint64_t = !clift.int<unsigned 8>

// CHECK: !my_commented_enum = !clift.enum<
// CHECK:   "/type-definition/0-EnumDefinition" as "my_commented_enum" : !uint64_t {
// CHECK:     "/enum-entry/0-EnumDefinition/0" as "enum_entry_my_commented_enum_0" : 0
// CHECK:       comment "Did I mention this value is zero?",
// CHECK:     "/enum-entry/0-EnumDefinition/1" as "enum_entry_my_commented_enum_1" : 1
// CHECK:       comment "And this one - is one!",
// CHECK:     "/enum-entry/0-EnumDefinition/18446744073709551615" as "enum_entry_my_commented_enum_18446744073709551615" : 18446744073709551615
// CHECK:       comment "And this one is too big for its own good!"
// CHECK:   }
// CHECK:   comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure enum-attached comments don't\0Aaccidentally get broken!"
// CHECK: >

!_type_definition_0_EnumDefinition = !clift.enum<
  "/type-definition/0-EnumDefinition" : !uint64_t {
    "/enum-entry/0-EnumDefinition/0" : 0,
    "/enum-entry/0-EnumDefinition/1" : 1,
    "/enum-entry/0-EnumDefinition/18446744073709551615" : 18446744073709551615
  }
>

module attributes {clift.module, clift.types = [!_type_definition_0_EnumDefinition]} {
}
