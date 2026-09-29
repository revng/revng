//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!void = !clift.void
!char = !clift.c_char<1>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes {clift.module} {
  clift.func @fun_0x40001004<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64",
    clift.legalized
  } {
    clift.expr {
      %0 = clift.str "" : !clift.const<!clift.array<1 x !char>>
      %1 = clift.addressof %0 : !clift.ptr<8 to !clift.const<!clift.array<1 x !char>>>
      clift.yield %1 : !clift.ptr<8 to !clift.const<!clift.array<1 x !char>>>
    }
  }
}
