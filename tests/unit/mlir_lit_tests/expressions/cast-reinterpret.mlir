//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!int32_t$const = !clift.primitive<const signed 4>
!int32_t$const$ptr = !clift.ptr<8 to !int32_t$const>

!uint32_t = !clift.primitive<unsigned 4>
!uint32_t$ptr = !clift.ptr<8 to !uint32_t>

!my_uint32 = !clift.defined<#clift.typedef<unique_handle = "/model-type/1",
                                           name = "my_uint32",
                                           underlying_type = !uint32_t>>

!my_enum = !clift.defined<#clift.enum<unique_handle = "/model-type/2",
                                      name = "my_enum",
                                      underlying_type = !uint32_t,
                                      fields = [
                                        <
                                          name = "enumerator",
                                          raw_value = 1
                                        >
                                      ]>>

%i = clift.undef : !int32_t
clift.cast<reinterpret> %i : !int32_t -> !uint32_t
clift.cast<reinterpret> %i : !int32_t -> !my_uint32
clift.cast<reinterpret> %i : !int32_t -> !my_enum

%t = clift.undef : !my_uint32
clift.cast<reinterpret> %t : !my_uint32 -> !int32_t

%e = clift.undef : !my_enum
clift.cast<reinterpret> %e : !my_enum -> !int32_t

%ip = clift.undef : !int32_t$ptr
clift.cast<reinterpret> %ip : !int32_t$ptr -> !uint32_t$ptr
clift.cast<reinterpret> %ip : !int32_t$ptr -> !int32_t$const$ptr

%icp = clift.undef : !int32_t$const$ptr
clift.cast<reinterpret> %icp : !int32_t$const$ptr -> !int32_t$ptr
