//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 1,
  fields = [<
      offset = 0,
      name = "",
      type = !clift.pointer<
        pointer_size = 8,
        pointee_type = !clift.defined<#clift.struct<id = 1>>
      >
    >
  ]
>>

clift.module {
} {
  s = !s
}
