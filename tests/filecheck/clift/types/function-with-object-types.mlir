//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !int32_t(!int32_t)
>
