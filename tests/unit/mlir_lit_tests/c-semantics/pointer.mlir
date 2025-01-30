//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 8,
  fields = [
    <
      offset = 1,
      name = "",
      type = !clift.pointer<
        pointer_size = 4,
        pointee_type = !clift.primitive<VoidKind 0>>
    >
  ]>>

clift.module {
  // CHECK: Pointer type is not representable in the target implementation.
  clift.global !s @x
}
