//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe import-descriptive-info %S/../0-import-types/TypedefType.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!float96_t = !clift.float<12>

// CHECK: !my_float = !clift.typedef<
// CHECK:   "/type-definition/0-TypedefDefinition" as "my_float" : !float96_t
// CHECK:   comment "Take a look at struct and function comment tests for more \22meat\22.\0A\0AThis one is just to ensure typedef-attached comments don't\0Aaccidentally get broken!"
// CHECK: >

!_type_definition_0_TypedefDefinition = !clift.typedef<
  "/type-definition/0-TypedefDefinition" : !float96_t
>

module attributes {clift.module, clift.types = [!_type_definition_0_TypedefDefinition]} {
}
