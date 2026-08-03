//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// The emitted C is reformatted with clang-format after emission is complete.
// The backend emits `//` comments with no space after the slashes, which
// clang-format normalizes. The plain path formats the C directly, while the
// PTML path strips the tags, formats, and maps the edits back; both must
// produce the same formatted C, which also checks that the PTML tags survive
// reformatting.

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes { clift.module } {
  // CHECK: void fun_0x40001004(void) {
  clift.func @fun_0x40001004<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64"
  } {
    // CHECK: // normalized by clang-format
    // CHECK: (uint32_t) 0;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.bitcast %0 : !int32_t -> !uint32_t
      clift.yield %1 : !uint32_t
    } attributes { clift.comments = ["normalized by clang-format"] }
  // CHECK: }
  }
}
