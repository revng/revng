//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c=inline-stack-frame %s -o /dev/null | FileCheck %s --check-prefix=INLINE
// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s --check-prefix=PLAIN

!void = !clift.void
!int32 = !clift.int<signed 4>

!frame_inner = !clift.struct<
  "/type-definition/0-StructDefinition" as "frame_inner" : size(8) {
    "/struct-field/0-StructDefinition/0" as "a" : offset(0) !int32,
    "/struct-field/0-StructDefinition/4" as "b" : offset(4) !int32
  }
>

// A typedef of the struct, itself wrapped in another typedef. Both must be
// stripped for `emitClassDefinition` to inline the underlying struct.
!frame_typedef_inner = !clift.typedef<
  "/type-definition/2-TypedefDefinition" as "frame_typedef_inner"
    : !frame_inner
>
!frame = !clift.typedef<
  "/type-definition/3-TypedefDefinition" as "frame"
    : !frame_typedef_inner
>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x1000:Code_x86_64"
  } {
    // INLINE:      struct _PACKED _SIZE(8) frame_inner {
    // INLINE-NEXT:   int32_t a _STARTS_AT(0);
    // INLINE-NEXT:   int32_t b _STARTS_AT(4);
    // INLINE-NEXT: } frame_var;
    //
    // PLAIN: frame frame_var;
    %frame_l = clift.local : !frame attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64",
      name = "frame_var",
      clift.stack_frame = true
    }
  }
}
