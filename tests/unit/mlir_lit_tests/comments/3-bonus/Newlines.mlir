//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-type-and-global-header=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void

// CHECK: /// This
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///  comment has
// CHECK: ///
// CHECK: ///
// CHECK: ///   extra
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: ///
// CHECK: /// awkward
// CHECK: ///
// CHECK: ///         spacing!
// CHECK: typedef void bonus;

!bonus = !clift.typedef<
  "/type-definition/0-TypedefDefinition" as "bonus" : !void
  comment "This\0A\0A\0A\0A\0A\0A\0A comment has\0A\0A\0A  extra\0A\0A\0A\0A\0A\0Aawkward\0A\0A        spacing!"
>

module attributes {clift.module, clift.types = [!bonus]} {
}
