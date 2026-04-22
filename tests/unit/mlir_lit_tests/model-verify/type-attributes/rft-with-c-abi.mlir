//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void

// CHECK: `_ABI` attribute value ('AAPCS64') differs from the model value ('raw_aarch64'). See '/type-definition/3-RawFunctionDefinition'

!f_3 = !clift.func<
  "/type-definition/3-RawFunctionDefinition" : !void()
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>
  ]
>

module attributes { clift.module, clift.types = [ !f_3 ] } {}
