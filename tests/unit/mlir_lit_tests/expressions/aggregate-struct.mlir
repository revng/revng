//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.const<!clift.primitive<signed 4>>

!s = !clift.struct<
  "" : size(8) {
    offset(0) as "x" : !int32_t,
    offset(4) as "y" : !int32_t
  }
>

%0 = clift.undef : !int32_t
%a = clift.aggregate(%0, %0) : !s

%1 = clift.undef : !int32_t$const
%b = clift.aggregate(%1 : !int32_t$const, %1 : !int32_t$const) : !s
