//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

// CHECK: field types must be object types
!s = !clift.defined<#clift.struct<
  unique_handle = "/model-type/1",
  name = "",
  size = 1,
  fields = [
    <
      offset = 0,
      name = "",
      type = !f
    >
  ]
>>
