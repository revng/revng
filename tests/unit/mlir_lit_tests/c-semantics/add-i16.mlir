//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!int16_t = !clift.primitive<signed 2>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !clift.primitive<void 0>,
  argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    %1 = clift.local !int16_t "x"

    // CHECK: causes integer promotion in the target implementation
    clift.expr {
      %2 = clift.add %1, %1 : !int16_t
      clift.yield %2 : !int16_t
    }
  }
}
