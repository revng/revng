//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!void = !clift.void

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>
