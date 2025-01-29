//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!void = !clift.primitive<VoidKind 0>
!function = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !void,
  argument_types = []>>

!function$ptr = !clift.pointer<pointee_type = !function, pointer_size = 8>

%function = clift.undef : !function
clift.cast<decay> %function : !function -> !function$ptr
