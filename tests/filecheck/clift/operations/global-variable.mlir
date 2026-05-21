//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

clift.global @x : !int32_t

clift.global @y : !int32_t = {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
}
