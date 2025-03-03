//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!s = !clift.defined<#clift.struct<
  unique_handle = "/model-type/1",
  name = "",
  size = 8,
  fields = [<
      offset = 0,
      name = "",
      type = !clift.ptr<
        8 to !clift.defined<#clift.struct<unique_handle = "/model-type/1">>
      >
    >
  ]
>>

clift.module {
} {
  s = !s
}
