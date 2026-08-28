//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!generic64_t = !clift.int<generic 8>
!generic64_t$ptr = !clift.ptr<8 to !generic64_t>

!int64_t = !clift.int<signed 8>
!int64_t$ptr = !clift.ptr<8 to !int64_t>

!uint64_t = !clift.int<unsigned 8>
!uint64_t$ptr = !clift.ptr<8 to !uint64_t>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.local : !generic64_t
    %0 = clift.local : !generic64_t
    // CHECK: clift.expr {
    clift.expr {
      %1 = clift.addressof %0 : !generic64_t$ptr
      %2 = clift.bitcast %1 : !generic64_t$ptr -> !int64_t$ptr
      %3 = clift.indirection %2 : !int64_t$ptr
      // CHECK: %1 = clift.imm 0 : !generic64_t
      %4 = clift.imm 0 : !int64_t
      // CHECK: %2 = clift.assign %0, %1 : !generic64_t
      %5 = clift.assign %3, %4 : !int64_t
      // CHECK: %3 = clift.bitcast %2 : !generic64_t -> !uint64_t
      %6 = clift.bitcast %5 : !int64_t -> !uint64_t
      clift.yield %6 : !uint64_t
    }
    // CHECK: }
  }
}
