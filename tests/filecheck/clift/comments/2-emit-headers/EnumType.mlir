//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-type-and-global-header=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!uint64_t = !clift.int<unsigned 8>

// CHECK: typedef enum _PACKED my_commented_enum my_commented_enum;

// CHECK: /// Take a look at struct and function comment tests for more "meat".
// CHECK: ///
// CHECK: /// This one is just to ensure enum-attached comments don't
// CHECK: /// accidentally get broken!
// CHECK: enum _ENUM_UNDERLYING(uint64_t) _PACKED my_commented_enum {
//
// CHECK:   /// Did I mention this value is zero?
// CHECK:   enum_entry_my_commented_enum_0 = 0x0U,
//
// CHECK:   /// And this one - is one!
// CHECK:   enum_entry_my_commented_enum_1 = 0x1U,
//
// CHECK:   /// And this one is too big for its own good!
// CHECK:   enum_entry_my_commented_enum_18446744073709551615 = 0xFFFFFFFFFFFFFFFFU,
// CHECK: };

!my_commented_enum = !clift.enum<
  "/type-definition/0-EnumDefinition" as "my_commented_enum" : !uint64_t {
    "/enum-entry/0-EnumDefinition/0" as "enum_entry_my_commented_enum_0" : 0
      comment "Did I mention this value is zero?",
    "/enum-entry/0-EnumDefinition/1" as "enum_entry_my_commented_enum_1" : 1
      comment "And this one - is one!",
    "/enum-entry/0-EnumDefinition/18446744073709551615" as "enum_entry_my_commented_enum_18446744073709551615" : 18446744073709551615
      comment "And this one is too big for its own good!"
  }
  comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure enum-attached comments don't\0Aaccidentally get broken!"
>

module attributes {clift.module, clift.types = [!my_commented_enum]} {
}
