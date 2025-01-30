//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

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

%0 = clift.undef : !s
%1 = clift.access<0> %0 : !s -> !int32_t

// CHECK: operand must be an lvalue-expression
%2 = clift.addressof %1 : !int32_t$ptr
