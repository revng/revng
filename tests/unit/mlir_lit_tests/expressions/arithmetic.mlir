//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

%1 = clift.local : !int32_t

clift.neg %1 : !int32_t
clift.add %1, %1 : !int32_t
clift.sub %1, %1 : !int32_t
clift.mul %1, %1 : !int32_t
clift.div %1, %1 : !int32_t
clift.rem %1, %1 : !int32_t

clift.not %1 : !int32_t -> !int32_t
clift.and %1, %1 : !int32_t -> !int32_t
clift.or %1, %1 : !int32_t -> !int32_t

clift.bitnot %1 : !int32_t
clift.bitand %1, %1 : !int32_t
clift.bitor %1, %1 : !int32_t
clift.bitxor %1, %1 : !int32_t

clift.shl %1, %1 : !int32_t
clift.shr %1, %1 : !int32_t

clift.eq %1, %1 : !int32_t -> !int32_t
clift.ne %1, %1 : !int32_t -> !int32_t
clift.lt %1, %1 : !int32_t -> !int32_t
clift.gt %1, %1 : !int32_t -> !int32_t
clift.le %1, %1 : !int32_t -> !int32_t
clift.ge %1, %1 : !int32_t -> !int32_t

clift.inc %1 : !int32_t
clift.dec %1 : !int32_t

clift.post_inc %1 : !int32_t
clift.post_dec %1 : !int32_t
