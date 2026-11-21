//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

// CHECK: #c_dialect = #clift.c_dialect<
#c_dialect = #clift.c_dialect<
  // CHECK: ImplicitQualifierDiscardingConversions = true,
  ImplicitQualifierDiscardingConversions = true,
  // CHECK-NOT: ImplicitPointerSignConversions = false
  ImplicitPointerSignConversions = false,
  // CHECK: ImplicitVoidPointerConversions = true
  ImplicitVoidPointerConversions = true
  // CHECK-NOT: ,
>
// CHECK: >

// CHECK: module attributes {clift.c_dialect = #c_dialect} {
module attributes {clift.c_dialect = #c_dialect} {
// CHECK: }
}
