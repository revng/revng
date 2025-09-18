//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!int32_t$const = !clift.const<!clift.primitive<signed 4>>
!int32_t$const$ptr = !clift.ptr<8 to !int32_t$const>

!uint32_t = !clift.primitive<unsigned 4>
!uint32_t$ptr = !clift.ptr<8 to !uint32_t>

!my_uint32 = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "my_uint32" : !uint32_t
>

!my_enum = !clift.enum<
  "/type-definition/2-EnumDefinition" as "my_enum" : !uint32_t {
    "/enum-entry/2-EnumDefinition/1" as "enumerator" : 1
  }
>

%i = clift.undef : !int32_t
clift.cast<bitcast> %i : !int32_t -> !uint32_t
clift.cast<bitcast> %i : !int32_t -> !my_uint32
clift.cast<bitcast> %i : !int32_t -> !my_enum

%t = clift.undef : !my_uint32
clift.cast<bitcast> %t : !my_uint32 -> !int32_t

%e = clift.undef : !my_enum
clift.cast<bitcast> %e : !my_enum -> !int32_t

%ip = clift.undef : !int32_t$ptr
clift.cast<bitcast> %ip : !int32_t$ptr -> !uint32_t$ptr
clift.cast<bitcast> %ip : !int32_t$ptr -> !int32_t$const$ptr

%icp = clift.undef : !int32_t$const$ptr
clift.cast<bitcast> %icp : !int32_t$const$ptr -> !int32_t$ptr
