//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void

// CHECK: `_ABI` attribute must have exactly one argument. See '/type-definition/2-CABIFunctionDefinition'

!f_2 = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !void()
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [
      #clift.identifier<"AAPCS64">,
      42
    ]>
  ]
>

module attributes { clift.module, clift.types = [ !f_2 ] } {}
