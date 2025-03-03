//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<void 0>
!function = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

!function$ptr = !clift.ptr<8 to !function>

%function = clift.undef : !function
clift.cast<decay> %function : !function -> !function$ptr
