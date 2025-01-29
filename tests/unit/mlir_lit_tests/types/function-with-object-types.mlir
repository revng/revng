//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1000",
  name = "f",
  return_type = !int32_t,
  argument_types = [!int32_t]>>
