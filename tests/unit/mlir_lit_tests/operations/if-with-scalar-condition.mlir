//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!float = !clift.primitive<float 4>
!pointer = !clift.ptr<8 to !int32_t>

!enum = !clift.enum<
  "/type-definition/1-EnumDefinition" : !int32_t {
    0
  }
>

clift.if {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
} {}

clift.if {
  %0 = clift.undef : !float
  clift.yield %0 : !float
} {}

clift.if {
  %0 = clift.undef : !pointer
  clift.yield %0 : !pointer
} {}

clift.if {
  %0 = clift.undef : !enum
  clift.yield %0 : !enum
} {}
