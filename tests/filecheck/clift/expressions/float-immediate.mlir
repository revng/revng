//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s
// RUN: %root/bin/revng clift-opt %s | %root/bin/revng clift-opt | FileCheck %s

!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!my_float = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "my_float" : !float32_t
>

// CHECK: clift.fimm 0.0 : !float32_t
clift.fimm 0.0 : !float32_t

// CHECK: clift.fimm 1.0 : !float32_t
clift.fimm 1.0 : !float32_t

// CHECK: clift.fimm 100.0 : !float32_t
clift.fimm 100.0 : !float32_t

// CHECK: clift.fimm 1.5 : !float32_t
clift.fimm 1.5 : !float32_t

// CHECK: clift.fimm 1.0E+10 : !float32_t
clift.fimm 1.0e10 : !float32_t

// CHECK: clift.fimm 0.100000001 : !float32_t
clift.fimm 0.1 : !float32_t

// CHECK: clift.fimm 16777216.0 : !float32_t
clift.fimm 16777217.0 : !float32_t

// CHECK: clift.fimm 3.40282347E+38 : !float32_t
clift.fimm 3.4028234663852886e38 : !float32_t

// CHECK: clift.fimm 1.40129846E-45 : !float32_t
clift.fimm 1.4012984643248171e-45 : !float32_t

// CHECK: clift.fimm 1.0 : !float64_t
clift.fimm 1.0 : !float64_t

// CHECK: clift.fimm 0.10000000000000001 : !float64_t
clift.fimm 0.1 : !float64_t

// CHECK: clift.fimm 123456789.0 : !float64_t
clift.fimm 123456789.0 : !float64_t

// CHECK: clift.fimm 1.7976931348623157E+308 : !float64_t
clift.fimm 1.7976931348623157e308 : !float64_t

// CHECK: clift.fimm 4.9406564584124654E-324 : !float64_t
clift.fimm 4.9406564584124654e-324 : !float64_t

// CHECK: clift.fimm 1.5 : !my_float
clift.fimm 1.5 : !my_float
