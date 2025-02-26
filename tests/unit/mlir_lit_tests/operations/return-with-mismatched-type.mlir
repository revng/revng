//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!int64_t = !clift.primitive<SignedKind 8>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1",
  name = "f",
  return_type = !int32_t,
  argument_types = [!int64_t]>>

clift.module {
  clift.func @f<!f>(%arg0 : !int64_t) {
    // CHECK: type does not match the function return type
    clift.return {
      clift.yield %arg0 : !int64_t
    }
  }
}
