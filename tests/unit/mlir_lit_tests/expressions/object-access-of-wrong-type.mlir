//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!uint32_t = !clift.primitive<UnsignedKind 4>

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 1,
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
