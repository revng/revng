//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>

!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f = !clift.func<"/model-type/1001" : !void(!int32_t$ptr, !generic64_t)>

module attributes {clift.module} {
  // x[0] -> *x
  clift.func @f<!f>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.imm 0 : !generic64_t
      // CHECK-NOT: clift.subscript
      %1 = clift.subscript %arg0, %0 : (!int32_t$ptr, !generic64_t)
      // CHECK: clift.indirection %arg0
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }

  // &(x[0]) -> x  (via composition with the `&*x -> x` rewriting)
  clift.func @g<!f>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.imm 0 : !generic64_t
      // CHECK-NOT: clift.subscript
      %1 = clift.subscript %arg0, %0 : (!int32_t$ptr, !generic64_t)
      // CHECK-NOT: clift.addressof
      %2 = clift.addressof %1 : !int32_t$ptr
      // CHECK: clift.yield %arg0
      clift.yield %2 : !int32_t$ptr
    }
    // CHECK: }
  }

  // x[i] is NOT rewritten (index is not a constant zero)
  clift.func @h<!f>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: clift.subscript
      %0 = clift.subscript %arg0, %arg1 : (!int32_t$ptr, !generic64_t)
      clift.yield %0 : !int32_t
    }
    // CHECK: }
  }

  // x[1] is NOT rewritten (constant index is not zero)
  clift.func @i<!f>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.imm 1 : !generic64_t
      // CHECK: clift.subscript
      %1 = clift.subscript %arg0, %0 : (!int32_t$ptr, !generic64_t)
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}
