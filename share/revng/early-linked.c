/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdint.h>

#include "revng/Runtime/PlainMetaAddress.h"

#include "support.h"

// The only purpose of this function is keeping alive the references to some
// symbols that are needed by revng
intptr_t ignore(void);
intptr_t ignore(void) {
  return (intptr_t) &saved_registers + (intptr_t) &setjmp
         + (intptr_t) &jmp_buffer + (intptr_t) &is_executable
         + (intptr_t) &unknownPC + (intptr_t) &_abort
         + (intptr_t) &_unreachable;
}

PlainMetaAddress build_PlainMetaAddress(uint32_t Epoch,
                                        uint16_t AddressSpace,
                                        uint16_t Type,
                                        uint64_t Address) {
  PlainMetaAddress Result;
  Result.Epoch = Epoch;
  Result.AddressSpace = AddressSpace;
  Result.Type = Type;
  Result.Address = Address;
  return Result;
}
