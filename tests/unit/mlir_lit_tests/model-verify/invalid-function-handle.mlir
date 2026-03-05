//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

module attributes {clift.module} {
  // CHECK: function with invalid handle: '/made-up-kind/something'
  clift.func @f<!f>() attributes {
    handle = "/made-up-kind/something"
  }
}
