//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>
!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>
!int32_t$const$ptr = !clift.pointer<pointee_type = !int32_t$const, pointer_size = 8>

!uint32_t = !clift.primitive<UnsignedKind 4>
!uint32_t$ptr = !clift.pointer<pointee_type = !uint32_t, pointer_size = 8>

!my_uint32 = !clift.defined<#clift.typedef<id = 1,
                                           name = "my_uint32",
                                           underlying_type = !uint32_t>>

!my_enum = !clift.defined<#clift.enum<id = 2,
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
