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
    // CHECK: %1 = clift.local : !int64_t
    %1 = clift.local : !int64_t
    // CHECK: %2 = clift.local : !uint64_t
    %2 = clift.local : !uint64_t
    // CHECK: clift.expr {
    clift.expr {
      %3 = clift.addressof %0 : !generic64_t$ptr
      %4 = clift.bitcast %3 : !generic64_t$ptr -> !int64_t$ptr
      %5 = clift.indirection %4 : !int64_t$ptr
      // CHECK: %3 = clift.bitcast %1 : !int64_t -> !generic64_t
      // CHECK: %4 = clift.assign %0, %3 : !generic64_t
      %6 = clift.assign %5, %1 : !int64_t

      %7 = clift.addressof %6 : !int64_t$ptr
      %8 = clift.bitcast %7 : !int64_t$ptr -> !uint64_t$ptr
      %9 = clift.indirection %8 : !uint64_t$ptr
      // CHECK: %5 = clift.bitcast %2 : !uint64_t -> !generic64_t
      // CHECK: %6 = clift.assign %4, %5 : !generic64_t
      %10 = clift.assign %9, %2 : !uint64_t

      // CHECK: clift.yield %6 : !generic64_t
      clift.yield %10 : !uint64_t
    }
    // CHECK: }
  }
}
