//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    clift.return {
      %x = clift.use @x : !int32_t
      clift.yield %x : !int32_t
    }
  }

  clift.global @x : !int32_t
}
