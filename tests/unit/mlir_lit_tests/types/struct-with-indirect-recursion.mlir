//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!s = !clift.defined<#clift.struct<"/model-type/1" : size(1) {
    offset(0) : !clift.defined<#clift.struct<"/model-type/2" : size(1) {
        offset(0) : !clift.defined<#clift.struct<"/model-type/1">>
      }
    >>
  }
>>

// CHECK: recursive class type
clift.module {
} {
  s = !s
}
