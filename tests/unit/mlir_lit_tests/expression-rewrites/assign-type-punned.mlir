//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!generic64_t = !clift.primitive<generic 8>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>

!f = !clift.func<"" : !void(!generic64_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !generic64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.addressof %arg0 : !generic64_t$ptr
      %1 = clift.bitcast %0 : !generic64_t$ptr -> !generic64_t$ptr
      %2 = clift.indirection %1 : !generic64_t$ptr
      // CHECK: %0 = clift.imm 0 : !generic64_t
      %3 = clift.imm 0 : !generic64_t
      // CHECK: %1 = clift.assign %arg0, %0 : !generic64_t
      %4 = clift.assign %2, %3 : !generic64_t
      // CHECK: clift.yield %1 : !generic64_t
      clift.yield %4 : !generic64_t
    }
    // CHECK: }
  }
}
