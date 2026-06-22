//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --refine-types --optimize-expressions | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!generic32_t = !clift.int<generic 4>

!f = !clift.func<"/model-type/1001" : !void(!generic32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !generic32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.bitcast %arg0 : !generic32_t -> !int32_t
      // CHECK: %1 = clift.bitcast %arg0 : !generic32_t -> !int32_t
      // CHECK: %2 = clift.add %0, %1 : !int32_t
      %0 = clift.add %arg0, %arg0 : !generic32_t
      %1 = clift.bitcast %0 : !generic32_t -> !int32_t
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %1 : !int32_t
    // CHECK: }
    }
  }
}
