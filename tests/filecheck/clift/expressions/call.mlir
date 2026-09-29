//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!int32_t$const = !clift.const<!clift.int<signed 4>>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void()
>

!g = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !void(!int32_t)
>

!h = !clift.func<
  "/type-definition/3-CABIFunctionDefinition" : !void(!int32_t$const)
>

%mi = clift.undef : !int32_t
%ci = clift.undef : !int32_t$const

%g = clift.undef : !clift.ptr<8 to !g>
%h = clift.undef : !clift.ptr<8 to !h>

clift.call %g(%mi) : !clift.ptr<8 to !g>
clift.call %g(%mi : !int32_t) : !clift.ptr<8 to !g>
clift.call %g(%ci : !int32_t$const) : !clift.ptr<8 to !g>

clift.call %h(%mi) : !clift.ptr<8 to !h>
clift.call %h(%mi : !int32_t) : !clift.ptr<8 to !h>
clift.call %h(%ci : !int32_t$const) : !clift.ptr<8 to !h>
