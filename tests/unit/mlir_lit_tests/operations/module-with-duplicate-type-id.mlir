//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!b = !clift.primitive<unsigned 1>

!s = !clift.defined<#clift.struct<unique_handle = "/model-type/1",
                                  name = "",
                                  size = 1,
                                  fields = []>>

!u = !clift.defined<#clift.union<unique_handle = "/model-type/1",
                                 name = "",
                                 fields = [<offset = 0, name = "", type = !b>]>>

// CHECK: two distinct type definitions with the same unique handle: '/model-type/1'
clift.module {
} {
  s = !s,
  u = !u
}
