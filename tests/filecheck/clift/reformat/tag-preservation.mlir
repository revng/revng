//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Each commented type definition is wrapped in a PTML region tag, and the
// header emitter separates consecutive definitions with a blank line. A comment
// whose body contains blank lines produces several of these in a row. When
// clang-format collapses such a run of blank lines, the run straddles the
// region's closing tag in the PTML; reformatting must preserve that tag rather
// than delete it as part of the whitespace. The PTML path going through
// `revng ptml` would fail to parse if a tag were dropped.

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-type-and-global-header=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

// CHECK: /// First typedef.
// CHECK: typedef void first;
// CHECK: /// Second typedef, with blank lines in its comment.
// CHECK: ///
// CHECK: ///
// CHECK: /// End of the comment.
// CHECK: typedef int32_t second;

!first = !clift.typedef<
  "/type-definition/0-TypedefDefinition" as "first" : !void
  comment "First typedef."
>

!second = !clift.typedef<
  "/type-definition/1-TypedefDefinition" as "second" : !int32_t
  comment "Second typedef, with blank lines in its comment.\0A\0A\0AEnd of the comment."
>

module attributes { clift.module, clift.types = [!first, !second] } {
}
