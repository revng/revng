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
      // CHECK: %1 = clift.implicit_cast %0 : !g -> !clift.ptr<8 to !g>
      %1 = clift.decay %0 : !g -> !clift.ptr<8 to !g>
      // CHECK: %2 = clift.implicit_cast %arg0 : !int32_t -> !uint8_t
      %2 = clift.truncate %arg0 : !int32_t -> !uint8_t
      // CHECK: %3 = clift.call %1(%2) : !clift.ptr<8 to !g>
      %3 = clift.call %1(%2) : !clift.ptr<8 to !g>
      // CHECK: clift.yield %3 : !void
      clift.yield %3 : !void
    }
    // CHECK: }
  }
}
