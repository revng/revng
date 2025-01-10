//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  id = 1,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.func @f<!f>() {
  %label = clift.make_label "label"
  clift.goto %label
  clift.assign_label %label
}
