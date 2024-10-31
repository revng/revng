//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>

clift.switch {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
} case 1 {
} case 0 {
} default {
}
