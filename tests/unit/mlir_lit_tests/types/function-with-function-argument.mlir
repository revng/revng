//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s

!void = !clift.primitive<VoidKind 0>

!f = !clift.defined<#clift.function<
  id = 1000,
  name = "f",
  return_type = !void,
  argument_types = []>>

!g = !clift.defined<#clift.function<
  id = 1001,
  name = "g",
  return_type = !void,
  argument_types = [!f]>>
