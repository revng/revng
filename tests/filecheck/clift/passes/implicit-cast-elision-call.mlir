//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --elide-implicit-casts | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" as "f" : !void(!int32_t)>
!g = !clift.func<"/model-type/1002" as "g" : !void(!uint8_t)>

module attributes {clift.module} {
  clift.func @g<!g>(%arg0 : !uint8_t) -> !void {}

  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.use @g : !g
      %0 = clift.use @g : !g
      // CHECK: %1 = clift.truncate %arg0 {clift.implicit} : !int32_t -> !uint8_t
      %1 = clift.truncate %arg0 : !int32_t -> !uint8_t
      // CHECK: %2 = clift.call %0(%1) : !g
      %2 = clift.call %0(%1) : !g
      // CHECK: clift.yield %2 : !void
      clift.yield %2 : !void
    }
    // CHECK: }
  }
}
