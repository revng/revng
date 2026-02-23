//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: ../%revngpipe emit-type-and-global-header /dev/null %s /dev/null /dev/stdout | ../%revngptml | FileCheck %s

!float96_t = !clift.primitive<float 12>

// CHECK: /// Take a look at struct and function comment tests for more "meat".
// CHECK: ///
// CHECK: /// This one is just to ensure typedef-attached comments don't
// CHECK: /// accidentally get broken!
// CHECK: typedef float96_t my_float;

!my_float = !clift.typedef<
  "/type-definition/0-TypedefDefinition" as "my_float" : !float96_t
  comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure typedef-attached comments don't\0Aaccidentally get broken!"
>

module attributes {clift.module, clift.types = [!my_float]} {
}
