//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t(!int32_t)
>

module attributes {clift.module} {
  clift.func @f<!f>(%arg : !int32_t) -> !int32_t {
    clift.return {
      %f = clift.use @f : !f
      %fp = clift.decay %f : !f -> !clift.ptr<8 to !f>
      %r = clift.call %fp(%arg) : !clift.ptr<8 to !f>
      clift.yield %r : !int32_t
    }
  }
}
