//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !clift.ptr<4 to !void>()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    clift.return {
      // CHECK: %0 = clift.null : !clift.ptr<8 to !void>
      %0 = clift.null : !clift.ptr<4 to !void>
      // CHECK: %1 = clift.ptr_resize %0 : !clift.ptr<8 to !void> -> !clift.ptr<4 to !void>
      // CHECK: clift.yield %1 : !clift.ptr<4 to !void>
      clift.yield %0 : !clift.ptr<4 to !void>
    }
  }
}
