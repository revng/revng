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

!plain = !clift.struct<
  "/type-definition/1-StructDefinition" as "plain" : size(4) {
    "/struct-field/1-StructDefinition/0" as "x" : offset(0) !int32
  }
>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x1000:Code_x86_64"
  } {
    // The stack-frame local gets its struct type inlined.
    //
    // INLINE:      struct _PACKED _SIZE(8) frame {
    // INLINE-NEXT:   int32_t a _STARTS_AT(0);
    // INLINE-NEXT:   int32_t b _STARTS_AT(4);
    // INLINE-NEXT: } frame_var;
    %frame = clift.local : !frame attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64",
      name = "frame_var",
      clift.stack_frame = true
    }

    // A regular local — no `clift.stack_frame`, not inlined even in INLINE mode.
    //
    // INLINE: plain regular_var;
    // PLAIN:  frame frame_var;
    // PLAIN:  plain regular_var;
    %regular = clift.local : !plain attributes {
      handle = "/local-variable/0x1000:Code_x86_64",
      name = "regular_var"
    }
  }
}
