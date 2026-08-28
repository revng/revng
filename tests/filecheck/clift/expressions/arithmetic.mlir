//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

%0 = clift.local : !int32_t
%1 = clift.test %0 : !int32_t

clift.neg %0 : !int32_t
clift.add %0, %0 : !int32_t
clift.sub %0, %0 : !int32_t
clift.mul %0, %0 : !int32_t
clift.sdiv %0, %0 : !int32_t
clift.udiv %0, %0 : !int32_t
clift.srem %0, %0 : !int32_t
clift.urem %0, %0 : !int32_t

clift.not %1
clift.and %1, %1
clift.or %1, %1

clift.bitnot %0 : !int32_t
clift.bitand %0, %0 : !int32_t
clift.bitor %0, %0 : !int32_t
clift.bitxor %0, %0 : !int32_t

clift.shl %0, %0 : !int32_t
clift.shr %0, %0 : !int32_t
clift.sar %0, %0 : !int32_t

clift.eq %0, %0 : !int32_t
clift.ne %0, %0 : !int32_t
clift.slt %0, %0 : !int32_t
clift.ult %0, %0 : !int32_t
clift.sgt %0, %0 : !int32_t
clift.ugt %0, %0 : !int32_t
clift.sle %0, %0 : !int32_t
clift.ule %0, %0 : !int32_t
clift.sge %0, %0 : !int32_t
clift.uge %0, %0 : !int32_t

clift.inc %0 : !int32_t
clift.dec %0 : !int32_t

clift.post_inc %0 : !int32_t
clift.post_dec %0 : !int32_t
