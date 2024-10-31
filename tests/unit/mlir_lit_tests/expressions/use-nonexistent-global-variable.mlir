//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<id = 1,
                                    name = "",
                                    return_type = !int32_t,
                                    argument_types = []>>

clift.module {
  clift.func @f<!f>() {
    clift.return {
      // CHECK: must reference a global variable or function
      %x = clift.use @x : !int32_t
      clift.yield %x : !int32_t
    }
  }
}
