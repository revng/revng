//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-type-and-global-header=ptml %s -o /dev/null | %revngptml | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: /// This struct has a lot of thoughtful and well argumented comments attached
// CHECK: /// to it by the user.
// CHECK: ///
// CHECK: /// It's extremely important we preserve these well and display them nicely.
// CHECK: ///
// CHECK: /// This comment is also already well-formatted, so there's no extra processing
// CHECK: /// to be done on it.
// CHECK: struct _PACKED _SIZE(8) my_commented_struct {
//
// CHECK:   /// This is the very first field, the comment for it is pretty short.
// CHECK:   uint8_t offset_0;
// CHECK:   uint8_t padding_at_1[1];
//
// CHECK:   /// This second field is extremely important, so the user put a whole lot of effort into this comment. It ended up really long. But it also turned out messy and mis-formatted! This line is so long, it's crazy! I wonder if we should help the user by introducing the extra formatting this so desperately needs!
// CHECK:   uint16_t offset_2;
//
// CHECK:   /// Unlike the previous one, this comment is short, but it contains a code
// CHECK:   /// snippet! So we should be careful and not mess this one up!
// CHECK:   /// ```cpp
// CHECK:   ///   int *a = nullopt;
// CHECK:   ///   *a = 42;
// CHECK:   /// ```
// CHECK:   ///
// CHECK:   /// (Your code can have no UB if you write nothing but comments!)
// CHECK:   ///
// CHECK:   /// P. S. This field is also twice the size of the other two!
// CHECK:   uint32_t offset_4;
// CHECK: };

!my_commented_struct = !clift.struct<
  "/type-definition/0-StructDefinition" as "my_commented_struct" : size(8) {
    "/struct-field/0-StructDefinition/0" as "offset_0" : offset(0) !uint8_t
      comment "This is the very first field, the comment for it is pretty short.",
    "/struct-field/0-StructDefinition/2" as "offset_2" : offset(2) !uint16_t
      comment "This second field is extremely important, so the user put a whole lot of effort into this comment. It ended up really long. But it also turned out messy and mis-formatted! This line is so long, it's crazy! I wonder if we should help the user by introducing the extra formatting this so desperately needs!",
    "/struct-field/0-StructDefinition/4" as "offset_4" : offset(4) !uint32_t
      comment "Unlike the previous one, this comment is short, but it contains a code\0Asnippet! So we should be careful and not mess this one up!\0A```cpp\0A  int *a = nullopt;\0A  *a = 42;\0A```\0A\0A(Your code can have no UB if you write nothing but comments!)\0A\0AP. S. This field is also twice the size of the other two!"
  }
  comment "This struct has a lot of thoughtful and well argumented comments attached\0Ato it by the user.\0A\0AIt's extremely important we preserve these well and display them nicely.\0A\0AThis comment is also already well-formatted, so there's no extra processing\0Ato be done on it."
>

module attributes {clift.module, clift.types = [!my_commented_struct]} {
}
