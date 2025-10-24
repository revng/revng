#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

class BinaryImporterHelper {
protected:
  model::Binary &Binary;
  uint64_t BaseAddress = 0;
  Logger<> &Logger;
  llvm::SmallVector<const model::Segment *> ExecutableSegments;
  bool SegmentsInitialized = false;

public:
  BinaryImporterHelper(model::Binary &Binary,
                       uint64_t BaseAddress,
                       ::Logger<> &Logger) :
    Binary(Binary), BaseAddress(BaseAddress), Logger(Logger) {}

public:
  MetaAddress relocate(MetaAddress Address) const {
    return Address += BaseAddress;
  }

  MetaAddress relocate(uint64_t Address) const {
    return relocate(fromGeneric(Address));
  }

  MetaAddress fromPC(uint64_t PC) const {
    using namespace model::Architecture;
    revng_assert(Binary.Architecture() != Invalid);
    return MetaAddress::fromPC(Binary.Architecture(), PC);
  }

  MetaAddress fromGeneric(uint64_t Address) const {
    using namespace model::Architecture;
    revng_assert(Binary.Architecture() != Invalid);
    return MetaAddress::fromGeneric(Binary.Architecture(), Address);
  }

public:
  void processSegments() {
    for (const model::Segment &Segment : Binary.Segments())
      if (Segment.IsExecutable())
        ExecutableSegments.push_back(&Segment);

    SegmentsInitialized = true;
  }

public:
  void registerExtraCodeAddress(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("register ExtraCodeAddress", Address);
      return;
    }

    Binary.ExtraCodeAddresses().insert(Address);
  }

  model::Function *registerFunctionEntry(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("register Function", Address);
      return nullptr;
    }

    return &Binary.Functions()[Address];
  }

  void setEntryPoint(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("set EntryPoint", Address);
      return;
    }

    Binary.EntryPoint() = Address;
  }

public:
  static uint64_t u64(uint64_t Value) { return Value; }

private:
  void report(const char *Action, const MetaAddress &Address) {
    revng_log(Logger,
              "Cannot " << Action << " " << Address.toString()
                        << " since it's not in an executable segment.");
  }

  /// \note Keep this private in order enforce not direct usage of EntryPoint
  ///       ExtraCodeAddress and new Functions registration.
  bool isExecutable(const MetaAddress &Address) const {
    revng_assert(SegmentsInitialized);
    auto ContainsAddress = [Address](const model::Segment *Segment) -> bool {
      return Segment->contains(Address);
    };
    return llvm::any_of(ExecutableSegments, ContainsAddress);
  }
};
