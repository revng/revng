//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>
!int32_t$array = !clift.array<1 x !int32_t>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!typedef$array = !clift.typedef<"" : !int32_t$array>

%array = clift.undef : !int32_t$array
clift.decay %array : !int32_t$array -> !int32_t$ptr

%typedef_array = clift.undef : !typedef$array
clift.decay %typedef_array : !typedef$array -> !int32_t$ptr
