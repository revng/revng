//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void

// CHECK: Unknown c-attribute ('_THIS_ONE_DOES_NOT_EXIST') found in '/type-definition/0-StructDefinition'

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}
  [
    #clift.c_attribute<"_THIS_ONE_DOES_NOT_EXIST" : "/macro/_THIS_ONE_DOES_NOT_EXIST">
  ]
>

module attributes { clift.module, clift.types = [ !s_0 ] } {}
