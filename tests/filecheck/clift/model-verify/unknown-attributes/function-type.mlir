//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

// CHECK: Unknown c-attribute ('_THIS_ONE_DOES_NOT_EXIST') found in '/type-definition/1-CABIFunctionDefinition'

!f_1 = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>,
    #clift.c_attribute<"_THIS_ONE_DOES_NOT_EXIST" : "/macro/_THIS_ONE_DOES_NOT_EXIST">
  ]
>

module attributes { clift.module, clift.types = [ !f_1 ] } {}
