//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | FileCheck %s

!int32_t = !clift.int<signed 4>

// CHECK: clift.local : !int32_t = ([[SELF:%arg[0-9]+]]) {
clift.local : !int32_t = (%0) {
  // CHECK: clift.yield [[SELF]] : !int32_t
  clift.yield %0 : !int32_t
// CHECK: }
}
