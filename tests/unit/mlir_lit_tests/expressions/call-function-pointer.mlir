//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int64_t = !clift.primitive<SignedKind 8>

!f = !clift.defined<#clift.function<unique_handle = "/model-type/1",
                                    name = "",
                                    return_type = !int32_t,
                                    argument_types = []>>

!g = !clift.defined<#clift.function<unique_handle = "/model-type/2",
                                    name = "",
                                    return_type = !int32_t,
                                    argument_types = [!int32_t]>>

!g$ptr = !clift.pointer<pointee_type = !g, pointer_size = 8>

clift.module {
  clift.func @f<!f>() {
    clift.return {
      %0 = clift.imm 0 : !int64_t
      %g = clift.cast<reinterpret> %0 : !int64_t -> !g$ptr
      %i = clift.imm 0 : !int32_t
      %r = clift.call %g(%i) : !g$ptr
      clift.yield %r : !int32_t
    }
  }
}
