//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void

// CHECK: `_ABI` attribute value ('raw_aarch64') differs from the model value ('AAPCS64'). See '/type-definition/2-CABIFunctionDefinition'

!f_2 = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !void()
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>
  ]
>

module attributes { clift.module, clift.types = [ !f_2 ] } {}
