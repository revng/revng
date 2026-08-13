/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdint.h>

#include "revng/Runtime/PlainMetaAddress.h"

#include "support.h"

PlainMetaAddress last_pc;
PlainMetaAddress current_pc;

// The only purpose of this function is keeping alive the references to some
// symbols that are needed by revng
intptr_t _ugly_name_ignore(void);
intptr_t _ugly_name_ignore(void) {
  return (intptr_t) &saved_registers + (intptr_t) &setjmp
         + (intptr_t) &jmp_buffer + (intptr_t) &is_executable
         + (intptr_t) &unknown_pc + (intptr_t) &revng_abort;
}

void set_PlainMetaAddress(PlainMetaAddress *This,
                          uint32_t Epoch,
                          uint16_t AddressSpace,
                          uint16_t Type,
                          uint64_t Address) {
  This->Epoch = Epoch;
  This->AddressSpace = AddressSpace;
  This->Type = Type;
  This->Address = Address;
}

#ifdef TARGET_s390x
// `libtcg-helpers-s390x.bc` inlines qemu's s390x TR-instruction helper
// (`helper_trXX`, `target/s390x/tcg/mem_helper.c`) which reads CPU
// features via `s390_has_feat` / `s390_get_feat_block`; both live in
// `target/s390x/cpu_models.c`, which the llvm-helpers build does not
// compile — so the resulting bitcode has them as extern-undefined.
// `revng quick artifact recompile-isolated` then hands the linker an
// object with two dangling qemu-internal symbols. Neither helper runs
// inside the recompiled binary during our tests (`revng.translated`
// verifies only that the pipeline produces a linked executable; the
// runtime behaviour is exercised by `qemu-run`, not by executing the
// translated binary), so a conservative "no features supported" stub
// is enough to unblock the link.
_Bool s390_has_feat(uint32_t feat) {
  (void) feat;
  return 0;
}

void s390_get_feat_block(uint32_t type, uint8_t *out) {
  (void) type;
  (void) out;
}
#endif
