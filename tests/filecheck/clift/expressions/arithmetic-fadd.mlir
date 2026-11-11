//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!my_float = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "my_float" : !float32_t
>

// CHECK: [[M:%[0-9]+]] = clift.undef : !float32_t
%m = clift.undef : !float32_t

// CHECK: [[C:%[0-9]+]] = clift.undef : !clift.const<!float32_t>
%c = clift.undef : !clift.const<!float32_t>

// CHECK: [[D:%[0-9]+]] = clift.undef : !float64_t
%d = clift.undef : !float64_t

// CHECK: [[T:%[0-9]+]] = clift.undef : !my_float
%t = clift.undef : !my_float

// CHECK: clift.fadd [[M]], [[M]] : !float32_t
clift.fadd %m, %m : !float32_t
// CHECK: clift.fadd [[M]], [[M]] : !float32_t
clift.fadd %m, %m : !float32_t -> !float32_t
// CHECK: clift.fadd [[M]], [[M]] : !float32_t
clift.fadd %m, %m : (!float32_t, !float32_t)
// CHECK: clift.fadd [[M]], [[M]] : !float32_t
clift.fadd %m, %m : (!float32_t, !float32_t) -> !float32_t

// CHECK: clift.fadd [[C]], [[M]] : (!clift.const<!float32_t>, !float32_t)
clift.fadd %c, %m : (!clift.const<!float32_t>, !float32_t)
// CHECK: clift.fadd [[C]], [[M]] : (!clift.const<!float32_t>, !float32_t)
clift.fadd %c, %m : (!clift.const<!float32_t>, !float32_t) -> !float32_t
// CHECK: clift.fadd [[M]], [[C]] : (!float32_t, !clift.const<!float32_t>)
clift.fadd %m, %c : (!float32_t, !clift.const<!float32_t>)
// CHECK: clift.fadd [[M]], [[C]] : (!float32_t, !clift.const<!float32_t>)
clift.fadd %m, %c : (!float32_t, !clift.const<!float32_t>) -> !float32_t

// CHECK: clift.fadd [[D]], [[D]] : !float64_t
clift.fadd %d, %d : !float64_t

// CHECK: clift.fadd [[T]], [[T]] : !my_float
clift.fadd %t, %t : !my_float
