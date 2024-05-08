#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/Architecture.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

class BinaryImporterHelper {
protected:
  model::Architecture::Values Architecture = model::Architecture::Invalid;
  uint64_t BaseAddress = 0;

public:
  BinaryImporterHelper(model::Architecture::Values Architecture,
                       uint64_t BaseAddress) :
    Architecture(Architecture), BaseAddress(BaseAddress) {}

public:
  MetaAddress relocate(MetaAddress Address) const {
    return Address += BaseAddress;
  }

  MetaAddress relocate(uint64_t Address) const {
    return relocate(fromGeneric(Address));
  }

  MetaAddress fromPC(uint64_t PC) const {
    using namespace model::Architecture;
    revng_assert(Architecture != Invalid);
    return MetaAddress::fromPC(toLLVMArchitecture(Architecture), PC);
  }

  MetaAddress fromGeneric(uint64_t Address) const {
    using namespace model::Architecture;
    revng_assert(Architecture != Invalid);
    return MetaAddress::fromGeneric(toLLVMArchitecture(Architecture), Address);
  }

public:
  static uint64_t u64(uint64_t Value) { return Value; }
};
