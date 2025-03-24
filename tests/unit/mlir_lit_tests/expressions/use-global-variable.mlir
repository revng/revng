//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<"/model-type/1" : !int32_t()>>

clift.module {
  clift.func @f<!f>() {
    clift.return {
      %x = clift.use @x : !int32_t
      clift.yield %x : !int32_t
    }
  }

  clift.global !int32_t @x
}
