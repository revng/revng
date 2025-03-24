//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

// CHECK: return type must be void or a non-array object type
!g = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "g",
  return_type = !f,
  argument_types = []>>
