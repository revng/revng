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
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.extend %arg0 {clift.implicit} : !uint8_t -> !int32_t
      %0 = clift.extend %arg0 : !uint8_t -> !int32_t
      // CHECK: %1 = clift.extend %arg0 {clift.implicit} : !uint8_t -> !int32_t
      %1 = clift.extend %arg0 : !uint8_t -> !int32_t
      // CHECK: %2 = clift.add %0, %1 : !int32_t
      %2 = clift.add %0, %1 : !int32_t
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %2 : !int32_t
    }
    // CHECK: }
  }
}
