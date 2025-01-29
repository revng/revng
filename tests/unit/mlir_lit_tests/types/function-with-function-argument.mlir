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

// CHECK: parameter type must be an object type
!g = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "g",
  return_type = !void,
  argument_types = [!f]>>
