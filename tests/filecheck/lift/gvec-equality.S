#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# Compare 20 bytes through two unaligned pointers. No source prototypes or
# debug information are provided to the decompiler.
#
# RUN: clang --target=x86_64-unknown-linux-gnu -nostdlib -static -Wl,--build-id=none -Wl,-T,%S/gvec-equality.ld %s -o %t
# RUN: %root/bin/revng quick analyze --analyses=parse-binary,detect-abi %t | FileCheck %s --check-prefix=ABI
# RUN: %root/bin/revng quick artifact emit-c-as-single-file %t -o %t.ptml
# RUN: %root/bin/revng ptml --plain %t.ptml | FileCheck %s --check-prefix=C --implicit-check-not=helper_gvec_
#
# ABI: Name: entry
# ABI: Definition: "/TypeDefinitions/[[PROTO:[0-9]+]]-RawFunctionDefinition"
# ABI: - ID: [[PROTO]]
# ABI: Arguments:
# ABI-NEXT: - Location: rsi_x86_64
# ABI: - Location: rdi_x86_64
# ABI-NOT: Location:
# ABI: ReturnValues:
# ABI-NEXT: - Location: rax_x86_64
# ABI-NOT: Location:
# ABI: PreservedRegisters:
#
# The unfixed output loses the left-hand loads, returns a constant comparison
# result, and reports an extra XMM return value. Check the recovered inputs and
# their comparisons directly, without compiling the decompiler's output.
# C-LABEL: generic64_t entry({{.*}}*argument_0, {{.*}}*argument_1)
# C: argument_0->offset_0 == (generic8_t) argument_1->offset_0
# C: argument_0->offset_8 == (generic8_t) argument_1->offset_8
# C: argument_0->offset_16 == (generic8_t) argument_1->offset_16
# C: return {{.*}}== 0xFFFFUL

.text
.globl entry
.type entry, @function
entry:
  movdqu (%rdi), %xmm0
  movdqu (%rsi), %xmm1
  pcmpeqb %xmm1, %xmm0
  movd 16(%rdi), %xmm2
  movd 16(%rsi), %xmm3
  pcmpeqb %xmm3, %xmm2
  pand %xmm2, %xmm0
  pmovmskb %xmm0, %eax
  cmpl $65535, %eax
  sete %al
  movzbl %al, %eax
  ret
.size entry, .-entry
.section .note.GNU-stack,"",@progbits
