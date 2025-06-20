//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

// CHECK: must have a case value for each case region
"clift.switch"() ({
  %0 = "clift.undef"() : () -> !clift.primitive<signed 4>
  "clift.yield"(%0) : (!clift.primitive<signed 4>) -> ()
}, {
}, { // Case region without matching case value.
}) {case_values = array<i64>} : () -> ()
