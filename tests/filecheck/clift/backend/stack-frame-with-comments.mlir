//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c=inline-stack-frame %s -o /dev/null | FileCheck %s --check-prefix=INLINE
// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s --check-prefix=PLAIN

!void = !clift.void
!int32 = !clift.int<signed 4>

// A struct with a struct-level Doxygen comment and per-field Doxygen
// comments. Used as a stack-frame local so the inline-stack-frame pass
// rewrites the declaration into an inlined struct definition — exercising
// all three comment surfaces (struct-level, per-field, local) in one go.
!frame = !clift.struct<
  "/type-definition/0-StructDefinition" as "frame" : size(8) {
    "/struct-field/0-StructDefinition/0" as "a" : offset(0) !int32
      comment "comment for field a",
    "/struct-field/0-StructDefinition/4" as "b" : offset(4) !int32
      comment "comment for field b"
  } comment "struct-level comment for the frame">

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x1000:Code_x86_64"
  } {
    // The stack-frame local gets its struct type inlined, with all three
    // comment surfaces preserved: the struct-level Doxygen comment, both
    // per-field Doxygen comments, and the local variable's `clift.comments`
    // list. The local comment is rendered before the struct definition
    // because the inlined emission is anchored by the local declaration.
    //
    // INLINE:      // variable comment
    // INLINE:      /// struct-level comment for the frame
    // INLINE-NEXT: struct _PACKED _SIZE(8) frame {
    // INLINE:   /// comment for field a
    // INLINE-NEXT:   int32_t a _STARTS_AT(0);
    // INLINE:   /// comment for field b
    // INLINE-NEXT:   int32_t b _STARTS_AT(4);
    // INLINE-NEXT: } frame_var;
    //
    // In PLAIN mode the struct definition is emitted into the header, so
    // only the local declaration ends up in the function body. The local
    // comment is still rendered in front of it.
    //
    // PLAIN:      // variable comment
    // PLAIN-NEXT: frame frame_var;
    %frame = clift.local : !frame attributes {
      handle = "/stack-frame-variable/0x1000:Code_x86_64",
      name = "frame_var",
      clift.stack_frame = true,
      clift.comments = ["variable comment"]
    }
  }
}
