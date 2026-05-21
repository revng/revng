//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null

!void = !clift.void

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>
!s_1 = !clift.struct<"/type-definition/1-StructDefinition" : size(64) {}
[#clift.c_attribute<"_CAN_CONTAIN_CODE" : "/macro/_CAN_CONTAIN_CODE">]>

!f_2 = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

!f_3 = !clift.func<
  "/type-definition/3-RawFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

module attributes { clift.module, clift.types = [ !s_0, !s_1, !f_2, !f_3 ] } {}
