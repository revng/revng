//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --elide-implicit-casts | FileCheck %s

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!uint8_t, !int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uint8_t, %arg1 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.truncate %arg1 {clift.implicit} : !int32_t -> !uint8_t
      %0 = clift.truncate %arg1 : !int32_t -> !uint8_t
      // CHECK: %1 = clift.assign %arg0, %0 : !uint8_t
      %1 = clift.assign %arg0, %0 : !uint8_t
      // CHECK: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }
    // CHECK: }
  }
}
