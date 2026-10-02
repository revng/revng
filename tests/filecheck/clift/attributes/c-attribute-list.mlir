//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s


#attribute_list = #clift.c_attribute_list<
  <"hello" : "/macro/hello" []>,
  <"foo" : "/macro/foo" [#clift.identifier<"bar">]>
>

// CHECK: #clift.c_attribute_list<
  // CHECK: <"hello" : "/macro/hello" []>
  // CHECK: ,
  // CHECK: <"foo" : "/macro/foo" [#clift.identifier<"bar">]>
// CHECK: >
module attributes { clift.c_attribute_list = #attribute_list } {}
