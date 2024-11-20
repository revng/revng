//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 1,
  fields = [
    <
      offset = 0,
      name = "",
      type = !clift.defined<#clift.struct<
        id = 2,
        name = "",
        size = 1,
        fields = [
          <
            offset = 0,
            name = "",
            type = !clift.defined<#clift.struct<id = 1>>
          >
        ]
      >>
    >
  ]
>>

// CHECK: recursive class type
clift.module {
} {
  s = !s
}
