//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>

!enum = !clift.defined<#clift.enum<
  unique_handle = "/model-type/1",
  name = "",
  underlying_type = !int32_t,
  fields = [
    <
      raw_value = 0,
      name = ""
    >
  ]
>>

clift.switch {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
}

clift.switch {
  %0 = clift.undef : !enum
  clift.yield %0 : !enum
}
