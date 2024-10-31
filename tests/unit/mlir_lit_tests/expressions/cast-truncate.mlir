//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int16_t = !clift.primitive<SignedKind 2>
!int32_t = !clift.primitive<SignedKind 4>

!uint16_t = !clift.primitive<UnsignedKind 2>
!uint32_t = !clift.primitive<UnsignedKind 4>

%i32 = clift.undef : !int32_t
%u32 = clift.undef : !uint32_t

clift.cast<truncate> %i32 : !int32_t -> !int16_t
clift.cast<truncate> %u32 : !uint32_t -> !uint16_t
