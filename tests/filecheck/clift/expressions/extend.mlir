//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>

!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

%i16 = clift.undef : !int16_t
%u16 = clift.undef : !uint16_t

clift.sext %i16 : !int16_t -> !int32_t
clift.zext %u16 : !uint16_t -> !uint32_t
