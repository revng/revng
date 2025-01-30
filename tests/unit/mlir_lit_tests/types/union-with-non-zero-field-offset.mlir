//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

// CHECK: union field offsets must be zero
!s = !clift.defined<#clift.union<
  unique_handle = "/model-type/1",
  name = "",
  fields = [
    <
      offset = 10,
      name = "",
      type = !clift.primitive<UnsignedKind 4>
    >
  ]
>>
