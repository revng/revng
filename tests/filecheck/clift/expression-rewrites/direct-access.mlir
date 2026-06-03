//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>

!s = !clift.union<
  "/type-definition/1-UnionDefinition" as "s" : {
    "/union-field/1-UnionDefinition/0" : !int32_t
  }
>

!f = !clift.func<"/model-type/1001" : !void(!s)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !s) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: clift.addressof
      %0 = clift.addressof %arg0 : !clift.ptr<8 to !s>
      // CHECK: %0 = clift.access< 0> %arg0 : !s -> !int32_t
      %1 = clift.access<indirect 0> %0 : !clift.ptr<8 to !s> -> !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}
