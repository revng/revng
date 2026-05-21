//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>]
>

module attributes {clift.module} {
  // CHECK: an isolated function with an invalid handle: '/function/0x40001002:Code_x86_64'
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001002:Code_x86_64"
  } {
  }
}
