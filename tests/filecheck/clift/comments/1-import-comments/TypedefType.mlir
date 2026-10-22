//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng pipeline run-pipe import-descriptive-info %S/../0-import-types/TypedefType.yml <(%pipe_input %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!float64_t = !clift.float<8>

// CHECK: !my_float = !clift.typedef<
// CHECK:   "/type-definition/0-TypedefDefinition" as "my_float" : !float64_t
// CHECK:   comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure typedef-attached comments don't\0Aaccidentally get broken!"
// CHECK: >

!_type_definition_0_TypedefDefinition = !clift.typedef<
  "/type-definition/0-TypedefDefinition" : !float64_t
>

module attributes {clift.module, clift.types = [!_type_definition_0_TypedefDefinition]} {
}
