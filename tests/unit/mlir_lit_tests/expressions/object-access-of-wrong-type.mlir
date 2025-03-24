//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 4>

!s = !clift.defined<#clift.struct<
  unique_handle = "/model-type/1",
  name = "",
  size = 4,
  fields = [<
      offset = 0,
      name = "x",
      type = !int32_t
    >
  ]
>>

%0 = clift.undef : !s

// CHECK: result type must match the selected member type
%1 = clift.access<0> %0 : !s -> !uint32_t
