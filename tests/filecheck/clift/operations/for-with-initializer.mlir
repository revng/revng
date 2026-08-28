//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

clift.for init : !int32_t {
  clift.local : !int32_t
} cond (%i) {
  %0 = clift.test %i : !int32_t
  clift.yield %0 : !clift.bool
} next (%i) {
  %0 = clift.inc %i : !int32_t
  clift.yield %0 : !int32_t
} body (%i) {
  clift.expr {
    clift.yield %i : !int32_t
  }
}
