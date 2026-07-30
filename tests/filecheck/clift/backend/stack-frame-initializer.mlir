//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c=inline-stack-frame %s -o /dev/null | FileCheck %s --check-prefix=INLINE
// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s --check-prefix=PLAIN

!void = !clift.void
!int32 = !clift.int<signed 4>

!frame = !clift.struct<
  "/type-definition/0-StructDefinition" as "frame" : size(8) {
    "/struct-field/0-StructDefinition/0" as "a" : offset(0) !int32,
    "/struct-field/0-StructDefinition/4" as "b" : offset(4) !int32
  }
>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x1000:Code_x86_64"
  } {
    // INLINE:      struct _PACKED _SIZE(8) frame {
    // INLINE-NEXT:   int32_t a _STARTS_AT(0);
    // INLINE-NEXT:   int32_t b _STARTS_AT(4);
    // INLINE-NEXT: } frame_var = {0, 0};
    //
    // PLAIN: frame frame_var = {0, 0};
    %frame = clift.local : !frame = {
      %z1 = clift.imm 0 : !int32
      %z2 = clift.imm 0 : !int32
      %agg = clift.aggregate(%z1, %z2) : !frame
      clift.yield %agg : !frame
    } attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64",
      name = "frame_var",
      clift.stack_frame = true
    }
  }
}
