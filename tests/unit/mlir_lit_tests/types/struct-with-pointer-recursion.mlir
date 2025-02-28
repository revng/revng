//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!s = !clift.defined<#clift.struct<"/model-type/1" : size(8) {
    offset(0) : !clift.ptr<8 to !clift.defined<#clift.struct<"/model-type/1">>>
  }
>>

clift.module {
} {
  s = !s
}
