//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | FileCheck %s

!int32_t = !clift.primitive<signed 4>

// CHECK: [[M:%[0-9]+]] = clift.undef : !int32_t
%m = clift.undef : !int32_t

// CHECK: [[C:%[0-9]+]] = clift.undef : !clift.const<!int32_t>
%c = clift.undef : !clift.const<!int32_t>

// CHECK: clift.add [[M]], [[M]]
clift.add %m, %m : !int32_t
// CHECK: clift.add [[M]], [[M]] : !int32_t
clift.add %m, %m : !int32_t -> !int32_t
// CHECK: clift.add [[M]], [[M]] : !int32_t
clift.add %m, %m : (!int32_t, !int32_t)
// CHECK: clift.add [[M]], [[M]] : !int32_t
clift.add %m, %m : (!int32_t, !int32_t) -> !int32_t

// CHECK: clift.add [[C]], [[M]] : (!clift.const<!int32_t>, !int32_t)
clift.add %c, %m : (!clift.const<!int32_t>, !int32_t)
// CHECK: clift.add [[C]], [[M]] : (!clift.const<!int32_t>, !int32_t)
clift.add %c, %m : (!clift.const<!int32_t>, !int32_t) -> !int32_t
// CHECK: clift.add [[M]], [[C]] : (!int32_t, !clift.const<!int32_t>)
clift.add %m, %c : (!int32_t, !clift.const<!int32_t>)
// CHECK: clift.add [[M]], [[C]] : (!int32_t, !clift.const<!int32_t>)
clift.add %m, %c : (!int32_t, !clift.const<!int32_t>) -> !int32_t
