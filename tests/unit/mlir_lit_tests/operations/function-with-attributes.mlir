//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1",
  name = "",
  return_type = !int32_t,
  argument_types = [!int32_t]>>

clift.func @f<!f>(!int32_t { my.a = "b" }) -> (!int32_t { my.c = "d" })
