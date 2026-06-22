//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --refine-types --optimize-expressions | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!generic32_t = !clift.int<generic 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: %0 = clift.local : !int32_t
    %0 = clift.local : !generic32_t

    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.assign %0, %arg0 : !int32_t
      %1 = clift.bitcast %arg0 : !int32_t -> !generic32_t
      %2 = clift.assign %0, %1 : !generic32_t
      // CHECK: clift.yield %1 : !int32_t
      clift.yield %2 : !generic32_t
    // CHECK: }
    }
  }
}
