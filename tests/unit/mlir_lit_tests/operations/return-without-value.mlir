//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" as "f" : !int32_t()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: must return a value in function not returning void
    clift.return {}
  }
}
