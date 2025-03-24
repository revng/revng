//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!float = !clift.primitive<float 4>
!pointer = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>

!enum = !clift.defined<#clift.enum<
  unique_handle = "/model-type/1",
  name = "",
  underlying_type = !int32_t,
  fields = [
    <
      raw_value = 0,
      name = ""
    >
  ]
>>

clift.for {} {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
} {} {}

clift.for {} {
  %0 = clift.undef : !float
  clift.yield %0 : !float
} {} {}

clift.for {} {
  %0 = clift.undef : !pointer
  clift.yield %0 : !pointer
} {} {}

clift.for {} {
  %0 = clift.undef : !enum
  clift.yield %0 : !enum
} {} {}
