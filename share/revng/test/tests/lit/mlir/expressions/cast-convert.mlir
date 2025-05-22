//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>

!float32_t = !clift.primitive<float 4>
!float64_t = !clift.primitive<float 8>

%i = clift.undef : !int32_t
%j = clift.undef : !int32_t$const
%f32 = clift.undef : !float32_t
%f64 = clift.undef : !float64_t

clift.cast<convert> %i : !int32_t -> !float32_t
clift.cast<convert> %j : !int32_t$const -> !float32_t
clift.cast<convert> %f32 : !float32_t -> !int32_t

clift.cast<convert> %i : !int32_t -> !float64_t
clift.cast<convert> %f64 : !float64_t -> !int32_t

clift.cast<convert> %f32 : !float32_t -> !float64_t
clift.cast<convert> %f64 : !float64_t -> !float32_t
