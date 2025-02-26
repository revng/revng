//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>

!s = !clift.defined<#clift.struct<
  unique_handle = "",
  name = "",
  size = 8,
  fields = [
    <
      offset = 0,
      name = "x",
      type = !int32_t
    >,
    <
      offset = 4,
      name = "y",
      type = !int32_t
    >
  ]
>>

%0 = clift.undef : !int32_t
%a = clift.aggregate(%0, %0) : !s

%1 = clift.undef : !int32_t$const
%b = clift.aggregate(%1 : !int32_t$const, %1 : !int32_t$const) : !s
