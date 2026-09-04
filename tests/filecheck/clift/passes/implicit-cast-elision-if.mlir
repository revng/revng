//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --elide-implicit-casts | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!uint8_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uint8_t) -> !void {
    // CHECK: clift.if {
    clift.if {
      // CHECK: %0 = clift.implicit_cast %arg0 : !uint8_t
      %0 = clift.test %arg0 : !uint8_t
      // CHECK: clift.yield %0 : !clift.bool
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
    // CHECK: }
    }
  }
}
