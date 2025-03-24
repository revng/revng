//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1",
  name = "f",
  return_type = !int32_t,
  argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    // CHECK: must return a value in function not returning void
    clift.return {}
  }
}
