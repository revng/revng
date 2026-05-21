//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" as "f" : !int32_t(!int64_t)
>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int64_t) {
    // CHECK: expression type must match the function return type
    clift.return {
      clift.yield %arg0 : !int64_t
    }
  }
}
