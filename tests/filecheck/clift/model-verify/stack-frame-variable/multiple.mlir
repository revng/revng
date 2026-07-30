//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!frame = !clift.struct<"/type-definition/0-StructDefinition" as "frame" : size(8) {}>
!f = !clift.func<"/type-definition/1-CABIFunctionDefinition" as "f" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>]>

// CHECK: More than one `stack_frame` local declared in '/function/0x1000:Code_x86_64'

module attributes { clift.module, clift.types = [ !frame, !f ] } {

  clift.func @f<!f>() -> !void attributes {
    handle = "/function/0x1000:Code_x86_64"
  } {
    %f0 = clift.local : !frame attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64",
      name = "frame_0",
      clift.stack_frame = true
    }

    %f1 = clift.local : !frame attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64/1",
      name = "frame_1",
      clift.stack_frame = true
    }
  }

}
