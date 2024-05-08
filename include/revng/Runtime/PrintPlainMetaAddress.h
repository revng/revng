#pragma once

/*
 * This file is distributed under the MIT License. See LICENSE.mit for details.
 */

#include <inttypes.h>
#include <stdio.h>

#include "revng/Runtime/PlainMetaAddress.h"

static int fprint_metaaddress(FILE *stream, PlainMetaAddress *address) {
  return fprintf(stream,
                 "{ 0x%" PRIx32 ", 0x%" PRIx16 ", 0x%" PRIx16 ", 0x%" PRIx64
                 " }\n",
                 address->Epoch,
                 address->AddressSpace,
                 address->Type,
                 address->Address);
}
