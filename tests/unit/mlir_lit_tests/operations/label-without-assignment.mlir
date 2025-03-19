//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1",
  name = "",
  return_type = !void,
  argument_types = []>>

clift.func @f<!f>() {
  // CHECK: clift.make_label with a use by clift.goto must have an assignment
  %label = clift.make_label "label"
  clift.goto %label
}
