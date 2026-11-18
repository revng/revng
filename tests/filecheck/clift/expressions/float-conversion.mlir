//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

!int8_t = !clift.int<signed 1>
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>

!uint32_t = !clift.int<unsigned 4>
!uint64_t = !clift.int<unsigned 8>

!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!my_float = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "my_float" : !float32_t
>

// CHECK: [[I8:%[0-9]+]] = clift.undef : !int8_t
%i8 = clift.undef : !int8_t
// CHECK: [[I32:%[0-9]+]] = clift.undef : !int32_t
%i32 = clift.undef : !int32_t
// CHECK: [[CI32:%[0-9]+]] = clift.undef : !clift.const<!int32_t>
%ci32 = clift.undef : !clift.const<!int32_t>
// CHECK: [[U32:%[0-9]+]] = clift.undef : !uint32_t
%u32 = clift.undef : !uint32_t
// CHECK: [[F32:%[0-9]+]] = clift.undef : !float32_t
%f32 = clift.undef : !float32_t
// CHECK: [[CF32:%[0-9]+]] = clift.undef : !clift.const<!float32_t>
%cf32 = clift.undef : !clift.const<!float32_t>
// CHECK: [[F64:%[0-9]+]] = clift.undef : !float64_t
%f64 = clift.undef : !float64_t
// CHECK: [[T:%[0-9]+]] = clift.undef : !my_float
%t = clift.undef : !my_float

// CHECK: clift.stof [[I32]] : !int32_t -> !float32_t
clift.stof %i32 : !int32_t -> !float32_t
// CHECK: clift.stof [[I8]] : !int8_t -> !float64_t
clift.stof %i8 : !int8_t -> !float64_t
// CHECK: clift.stof [[CI32]] : !clift.const<!int32_t> -> !float32_t
clift.stof %ci32 : !clift.const<!int32_t> -> !float32_t
// CHECK: clift.stof [[I32]] : !int32_t -> !my_float
clift.stof %i32 : !int32_t -> !my_float

// CHECK: clift.utof [[U32]] : !uint32_t -> !float32_t
clift.utof %u32 : !uint32_t -> !float32_t
// CHECK: clift.utof [[U32]] : !uint32_t -> !float64_t
clift.utof %u32 : !uint32_t -> !float64_t

// CHECK: clift.ftos [[F32]] : !float32_t -> !int32_t
clift.ftos %f32 : !float32_t -> !int32_t
// CHECK: clift.ftos [[F64]] : !float64_t -> !int8_t
clift.ftos %f64 : !float64_t -> !int8_t
// CHECK: clift.ftos [[CF32]] : !clift.const<!float32_t> -> !int64_t
clift.ftos %cf32 : !clift.const<!float32_t> -> !int64_t
// CHECK: clift.ftos [[T]] : !my_float -> !int32_t
clift.ftos %t : !my_float -> !int32_t

// CHECK: clift.ftou [[F32]] : !float32_t -> !uint32_t
clift.ftou %f32 : !float32_t -> !uint32_t
// CHECK: clift.ftou [[F64]] : !float64_t -> !uint64_t
clift.ftou %f64 : !float64_t -> !uint64_t

// CHECK: clift.stof [[U32]] : !uint32_t -> !float32_t
clift.stof %u32 : !uint32_t -> !float32_t
// CHECK: clift.utof [[I32]] : !int32_t -> !float32_t
clift.utof %i32 : !int32_t -> !float32_t
// CHECK: clift.ftos [[F32]] : !float32_t -> !uint32_t
clift.ftos %f32 : !float32_t -> !uint32_t
// CHECK: clift.ftou [[F32]] : !float32_t -> !int32_t
clift.ftou %f32 : !float32_t -> !int32_t

// CHECK: clift.ftof [[F32]] : !float32_t -> !float64_t
clift.ftof %f32 : !float32_t -> !float64_t
// CHECK: clift.ftof [[F64]] : !float64_t -> !float32_t
clift.ftof %f64 : !float64_t -> !float32_t
// CHECK: clift.ftof [[CF32]] : !clift.const<!float32_t> -> !float64_t
clift.ftof %cf32 : !clift.const<!float32_t> -> !float64_t
// CHECK: clift.ftof [[T]] : !my_float -> !float64_t
clift.ftof %t : !my_float -> !float64_t
