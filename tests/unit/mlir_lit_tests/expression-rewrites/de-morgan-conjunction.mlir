//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t, !int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.not %arg0 : !int32_t -> !int32_t
      %1 = clift.not %arg1 : !int32_t -> !int32_t
      // CHECK: %0 = clift.or %arg0, %arg1 : !int32_t -> !int32_t
      %2 = clift.and %0, %1 : !int32_t -> !int32_t
      %3 = clift.not %2 : !int32_t -> !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %3 : !int32_t
    // CHECK: }
    }
  }
}
