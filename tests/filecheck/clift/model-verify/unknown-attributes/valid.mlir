//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>

!f_1 = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

!f_2 = !clift.func<
  "/type-definition/2-RawFunctionDefinition" : !uint64_t(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

module attributes { clift.module, clift.types = [ !s_0, !f_1, !f_2 ] } {

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attributes = [],
      clift.handle = "/cabi-argument/2-RawFunctionDefinition/x0_aarch64"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

}
