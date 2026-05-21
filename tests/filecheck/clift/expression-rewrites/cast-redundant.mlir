//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!generic32_t = !clift.int<generic 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: clift.bitcast
      %0 = clift.bitcast %arg0 : !int32_t -> !int32_t
      // CHECK: clift.yield %arg0
      clift.yield %0 : !int32_t
    }
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[RESULT:%[A-Za-z0-9]*]] = clift.bitcast %{{[A-Za-z0-9]*}} : !int32_t -> !generic32_t
      // CHECK-NOT: clift.bitcast
      %0 = clift.bitcast %arg0 : !int32_t -> !uint32_t
      %1 = clift.bitcast %0 : !uint32_t -> !generic32_t
      // CHECK: clift.yield [[RESULT]]
      clift.yield %1 : !generic32_t
    }
    // CHECK: }
  }
}
