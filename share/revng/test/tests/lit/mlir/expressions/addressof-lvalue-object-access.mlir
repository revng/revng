//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!s = !clift.defined<#clift.struct<
  "/type-definition/1-StructDefinition" : size(4) {
    offset(0) as "x" : !int32_t
  }
>>
!p_s = !clift.ptr<8 to !s>

%0 = clift.undef : !p_s
%1 = clift.indirection %0 : !p_s
%2 = clift.access<0> %1 : !s -> !int32_t
%3 = clift.addressof %2 : !int32_t$ptr
