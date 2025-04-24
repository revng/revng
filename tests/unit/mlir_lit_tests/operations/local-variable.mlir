//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

clift.local !int32_t

clift.local !int32_t = {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
}
