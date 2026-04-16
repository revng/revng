//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>

!s = !clift.union<
  "/type-definition/1-UnionDefinition" as "s" : {
    "/union-field/1-UnionDefinition/0" : !int32_t
  }
>

!f = !clift.func<"/model-type/1001" : !void(!clift.ptr<8 to !s>)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !clift.ptr<8 to !s>) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NOT: clift.indirection
      %0 = clift.indirection %arg0 : !clift.ptr<8 to !s>
      // CHECK: %0 = clift.access<indirect 0> %arg0 : !clift.ptr<8 to !s> -> !int32_t
      %1 = clift.access<0> %0 : !s -> !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}
