//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(8) {
    offset(0) : !clift.ptr<8 to !clift.struct<"/type-definition/1-StructDefinition">>
  }
>

module attributes {clift.module, clift.test = !s} {}
