//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// A `clift.str`, whose characters are always `number8_t`, cast to the `uint8_t`
// a segment field holding a string has in the model. The pointee types differ,
// so the cast is not one C performs on its own and elision keeps it.

// RUN: %root/bin/revng clift-opt %s --elide-implicit-casts | FileCheck %s --check-prefix=IR
// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s --check-prefix=C
// RUN: %root/bin/revng clift-opt --elide-implicit-casts --emit-c %s -o /dev/null | FileCheck %s --check-prefix=C

!char$const = !clift.const<!clift.int<number 1>>
!uchar$const = !clift.const<!clift.int<unsigned 1>>

!str = !clift.array<6 x !char$const>

!char$const$ptr = !clift.ptr<8 to !char$const>
!uchar$const$ptr = !clift.ptr<8 to !uchar$const>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !uchar$const$ptr()
>

module attributes {clift.module} {
  // C: const uint8_t *fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // C: return (const uint8_t *) "hello";
    clift.return {
      %s = clift.str "hello" : !str
      %d = clift.decay %s : !str -> !char$const$ptr

      // An elided cast carries a `{clift.implicit}` attribute after the
      // operand, where this one has a `:`.
      // IR: clift.bitcast %{{[0-9]+}} : !clift.ptr<8 to !clift.const<!number8_t>> -> !clift.ptr<8 to !clift.const<!uint8_t>>
      %c = clift.bitcast %d : !char$const$ptr -> !uchar$const$ptr

      clift.yield %c : !uchar$const$ptr
    }
  }
  // C: }
  // IR-NOT: clift.implicit
}
