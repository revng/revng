//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t(!int32_t)
>

clift.func @f<!f>(!int32_t { my.a = "b" }) -> (!int32_t { my.c = "d" })
