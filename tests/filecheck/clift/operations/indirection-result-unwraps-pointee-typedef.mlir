//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// Reaching the pointee unwraps a typedef around the pointer, but not one
// around the pointee: a result that unwraps the pointee typedef is rejected.

!int32_t = !clift.int<signed 4>
!int_alias = !clift.typedef<
  "/type-definition/3-TypedefDefinition" as "int_alias" : !int32_t
>
!p_to_alias = !clift.ptr<8 to !int_alias>

%0 = clift.undef : !p_to_alias
// CHECK: 'clift.indirection' op failed to verify that pointer pointee type matches the result type
%1 = "clift.indirection"(%0) : (!p_to_alias) -> !int32_t
