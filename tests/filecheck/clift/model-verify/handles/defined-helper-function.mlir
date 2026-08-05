//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>]
>

module attributes {clift.module} {
  // CHECK: non-isolated function with a definition: '/helper-function/foo'
  clift.func @f<!f>() attributes {
    handle = "/helper-function/foo"
  } {
    ^0:
  }
}
