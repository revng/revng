//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --elide-implicit-casts | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!float32_t = !clift.float<4>

!f = !clift.func<"/model-type/1001" : !void()>

#c_dialect = #clift.c_dialect<ImplicitVoidPointerConversions = true>

module attributes {clift.module, clift.c_dialect = #c_dialect} {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.local : !clift.ptr<8 to !int32_t>
    %0 = clift.local : !clift.ptr<8 to !int32_t>

    // float32_t -> int32_t (incompatible types)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.undef : !clift.ptr<8 to !float32_t>
      %1 = clift.undef : !clift.ptr<8 to !float32_t>
      // CHECK: %2 = clift.bitcast %1 : !clift.ptr<8 to !float32_t> -> !clift.ptr<8 to !int32_t>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !float32_t> -> !clift.ptr<8 to !int32_t>
      // CHECK: %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      // CHECK: clift.yield %3 : !clift.ptr<8 to !int32_t>
      clift.yield %3 : !clift.ptr<8 to !int32_t>
    }
    // CHECK: }

    // uint32_t -> int32_t (signedness conversion)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.undef : !clift.ptr<8 to !uint32_t>
      %1 = clift.undef : !clift.ptr<8 to !uint32_t>
      // CHECK: %2 = clift.bitcast %1 : !clift.ptr<8 to !uint32_t> -> !clift.ptr<8 to !int32_t>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !uint32_t> -> !clift.ptr<8 to !int32_t>
      // CHECK: %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      // CHECK: clift.yield %3 : !clift.ptr<8 to !int32_t>
      clift.yield %3 : !clift.ptr<8 to !int32_t>
    }
    // CHECK: }

    // int32_t const -> int32_t (discards qualifiers)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.undef : !clift.ptr<8 to !clift.const<!int32_t>>
      %1 = clift.undef : !clift.ptr<8 to !clift.const<!int32_t>>
      // CHECK: %2 = clift.bitcast %1 : !clift.ptr<8 to !clift.const<!int32_t>> -> !clift.ptr<8 to !int32_t>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !clift.const<!int32_t>> -> !clift.ptr<8 to !int32_t>
      // CHECK: %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      // CHECK: clift.yield %3 : !clift.ptr<8 to !int32_t>
      clift.yield %3 : !clift.ptr<8 to !int32_t>
    }
    // CHECK: }

    // void -> int32_t (void to non-void) (elided)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %1 = clift.undef : !clift.ptr<8 to !void>
      %1 = clift.undef : !clift.ptr<8 to !void>
      // CHECK: %2 = clift.implicit_cast %1 : !clift.ptr<8 to !void> -> !clift.ptr<8 to !int32_t>
      %2 = clift.bitcast %1 : !clift.ptr<8 to !void> -> !clift.ptr<8 to !int32_t>
      // CHECK: %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      %3 = clift.assign %0, %2 : !clift.ptr<8 to !int32_t>
      // CHECK: clift.yield %3 : !clift.ptr<8 to !int32_t>
      clift.yield %3 : !clift.ptr<8 to !int32_t>
    }
    // CHECK: }
  }
}
