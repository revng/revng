//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe import-descriptive-info %S/../model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

#loc1 = loc("/instruction/0x40001004:Code_x86_64/0x40001004:Code_x86_64/0x40001005:Code_x86_64")

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001004<!fun_0x40001004_t>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001004:Code_x86_64"
  // CHECK: }
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64"
  } {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 0 : !int32_t
      %0 = clift.imm 0 : !int32_t loc(#loc1)
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: } attributes {
    // CHECK: clift.comments = ["hello"]
    // CHECK: }
    }
  }
}
