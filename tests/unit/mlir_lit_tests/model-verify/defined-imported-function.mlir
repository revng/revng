//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes {clift.module} {
  // CHECK: non-isolated function with a definition: '/dynamic-function/foo'
  clift.func @f<!f>() attributes {
    handle = "/dynamic-function/foo"
  } {
    ^0:
  }
}
