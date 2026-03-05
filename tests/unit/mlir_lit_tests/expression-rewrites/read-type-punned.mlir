//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!generic64_t = !clift.int<generic 8>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>

!uint64_t = !clift.int<unsigned 8>
!uint64_t$ptr = !clift.ptr<8 to !uint64_t>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.local : !generic64_t
    %0 = clift.local : !generic64_t
    // CHECK: %1 = clift.local : !uint64_t
    %1 = clift.local : !uint64_t
    // CHECK: clift.expr {
    clift.expr {
      %2 = clift.addressof %0 : !generic64_t$ptr
      %3 = clift.bitcast %2 : !generic64_t$ptr -> !uint64_t$ptr
      %4 = clift.indirection %3 : !uint64_t$ptr
      %5 = clift.assign %1, %4 : !uint64_t
      clift.yield %5 : !uint64_t
    }
    // CHECK: }
  }
}
