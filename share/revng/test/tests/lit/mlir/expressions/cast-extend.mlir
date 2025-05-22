//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int16_t = !clift.primitive<signed 2>
!int32_t = !clift.primitive<signed 4>

!uint16_t = !clift.primitive<unsigned 2>
!uint32_t = !clift.primitive<unsigned 4>

%i16 = clift.undef : !int16_t
%u16 = clift.undef : !uint16_t

clift.cast<extend> %i16 : !int16_t -> !int32_t
clift.cast<extend> %u16 : !uint16_t -> !uint32_t
