//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --deduce-immediate-radices | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) {
    // CHECK: clift.switch {
    clift.switch {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
    // CHECK: } attributes {clift.radix = 10 : ui32}
    }

    // CHECK: clift.switch {
    clift.switch {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } case 11185083 {
    } case 11185083 {
    // CHECK: } attributes {clift.radix = 16 : ui32}
    }

    // CHECK: clift.switch {
    clift.switch {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
    // CHECK: } case 11185083 {
    } case 11185083 {
    // CHECK: } attributes {clift.radix = 10 : ui32}
    }

    // CHECK: clift.switch {
    clift.switch {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
    // CHECK: } case 11185083 {
    } case 11185083 {
    // CHECK: } case 13422045 {
    } case 13422045 {
    // CHECK: } attributes {clift.radix = 16 : ui32}
    }
  }
}
