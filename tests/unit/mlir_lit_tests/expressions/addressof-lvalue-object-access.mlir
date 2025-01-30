//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 4,
  fields = [<
      offset = 0,
      name = "x",
      type = !int32_t
    >
  ]
>>
!p_s = !clift.pointer<pointee_type = !s, pointer_size = 8>

%0 = clift.undef : !p_s
%1 = clift.indirection %0 : !p_s
%2 = clift.access<0> %1 : !s -> !int32_t
%3 = clift.addressof %2 : !int32_t$ptr
