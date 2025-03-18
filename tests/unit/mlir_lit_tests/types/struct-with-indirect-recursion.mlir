//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!s = !clift.defined<#clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {
    offset(0) : !clift.defined<#clift.struct<
      "/type-definition/2-StructDefinition" : size(1) {
        offset(0) : !clift.defined<#clift.struct<"/type-definition/1-StructDefinition">>
      }
    >>
  }
>>

// CHECK: recursive class type
clift.module {
} {
  s = !s
}
