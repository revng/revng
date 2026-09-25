//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void
!char = !clift.c_char<1>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes {clift.module} {
  clift.func @fun_0x40001004<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64",
    clift.legalized
  } {
    // CHECK: C character types may only be used by address-of and string literal operations
    clift.expr {
      %0 = clift.undef : !clift.const<!clift.array<1 x !char>>
      clift.yield %0 : !clift.const<!clift.array<1 x !char>>
    }
  }
}
