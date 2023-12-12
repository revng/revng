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
intptr_t ignore(void);
intptr_t ignore(void) {
  return (intptr_t) &saved_registers + (intptr_t) &setjmp
         + (intptr_t) &jmp_buffer + (intptr_t) &is_executable
         + (intptr_t) &unknown_pc + (intptr_t) &_abort
         + (intptr_t) &_unreachable;
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
