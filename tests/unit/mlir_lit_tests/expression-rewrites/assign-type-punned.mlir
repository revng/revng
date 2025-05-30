//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!generic64_t = !clift.primitive<generic 8>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>

!uint64_t = !clift.primitive<unsigned 8>
!uint64_t$ptr = !clift.ptr<8 to !uint64_t>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    %x = clift.local !generic64_t

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.addressof %x : !generic64_t$ptr
      %1 = clift.cast<bitcast> %0 : !generic64_t$ptr -> !uint64_t$ptr
      %2 = clift.indirection %1 : !uint64_t$ptr
      %3 = clift.imm 0 : !uint64_t
      %4 = clift.assign %2, %3 : !uint64_t
      clift.yield %4 : !uint64_t
    }
    // CHECK: }
  }
}
