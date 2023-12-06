#pragma once

/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdint.h>

typedef struct {
  uint32_t Epoch;
  uint16_t AddressSpace;
  uint16_t Type;
  uint64_t Address;
} PlainMetaAddress;

// NOLINTNEXTLINE
void set_PlainMetaAddress(PlainMetaAddress *This,
                          uint32_t Epoch,
                          uint16_t AddressSpace,
                          uint16_t Type,
                          uint64_t Address);
