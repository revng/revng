#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/ImportLogger.h"
#include "revng/Support/Debug.h"
#include "revng/Support/LDDTree.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/MetaAddressRange.h"

class BinaryImporterHelper {
protected:
  TupleTree<model::Binary> &Binary;
  uint64_t BaseAddress = 0;
  Logger &Logger;
  MetaAddressRangeSet ExecutableRanges;
  bool SegmentsInitialized = false;

public:
  BinaryImporterHelper(TupleTree<model::Binary> &Binary,
                       uint64_t BaseAddress,
                       ::Logger &Logger) :
    Binary(Binary), BaseAddress(BaseAddress), Logger(Logger) {}

public:
  MetaAddress relocate(MetaAddress Address) const {
    return Address += BaseAddress;
  }

  MetaAddress relocate(uint64_t Address) const {
    return relocate(fromGeneric(Address));
  }

  MetaAddress toPC(const MetaAddress &Generic) const {
    revng_assert(Generic.isGeneric());
    using namespace model::Architecture;
    revng_assert(Binary->Architecture() != Invalid);
    return MetaAddress::fromPC(Binary->Architecture(),
                               Generic.address(),
                               Generic.epoch(),
                               Generic.addressSpace());
  }

  MetaAddress fromPC(uint64_t PC) const {
    using namespace model::Architecture;
    revng_assert(Binary->Architecture() != Invalid);
    return MetaAddress::fromPC(Binary->Architecture(), PC);
  }

  MetaAddress fromGeneric(uint64_t Address) const {
    using namespace model::Architecture;
    revng_assert(Binary->Architecture() != Invalid);
    return MetaAddress::fromGeneric(Binary->Architecture(), Address);
  }

public:
  void processSegments() {
    ExecutableRanges = Binary->executableRanges();
    SegmentsInitialized = true;
  }

public:
  void registerExtraCodeAddress(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("register ExtraCodeAddress", Address);
      return;
    }

    Binary->ExtraCodeAddresses().insert(Address);
  }

  model::Function *registerFunctionEntry(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("register Function", Address);
      return nullptr;
    }

    if (not Binary->Functions().contains(Address))
      revng_log(Logger, "Registering new function at " << Address.toString());

    return &Binary->Functions()[Address];
  }

  /// Similar to registerFunctionEntry but if another function at the same
  /// address already exists, return that one. This is necessary in certain
  /// situations where the raw address is available but it's not known its type
  /// (e.g., Thumb vs regular ARM).
  model::Function *matchFunctionEntry(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("match Function", Address);
      return nullptr;
    }

    llvm::SmallVector<MetaAddress> Candidates;
    Candidates.push_back(Address);

    using namespace MetaAddressType;
    for (auto CodeType : archCodeTypes(arch(Address.type()))) {
      if (CodeType == Address.type())
        continue;

      auto NewAddress = Address.replaceType(CodeType);
      if (NewAddress.isValid())
        Candidates.push_back(std::move(NewAddress));
    }

    unsigned Matches = 0;
    auto EndIt = Binary->Functions().end();
    auto ResultIt = EndIt;
    for (const auto &Candidate : Candidates) {
      auto MatchIt = Binary->Functions().find(Candidate);
      if (MatchIt != EndIt) {
        ++Matches;
        if (ResultIt == EndIt) {
          // Take the first match
          ResultIt = MatchIt;
        }
      }
    }

    if (ResultIt == EndIt) {
      revng_log(Logger,
                "Warning: no match for function at "
                  << Address.toString() << ". Creating it despite this.");
      return &Binary->Functions()[Address];
    } else if (Matches == 1 and ResultIt->Entry() != Address) {
      revng_log(Logger,
                "Matching " << ResultIt->Entry().toString() << " for "
                            << Address.toString());
      return &*ResultIt;
    } else if (Matches > 1 and ResultIt->Entry() != Address) {
      revng_log(Logger,
                "Warning: multiple matches for "
                  << Address.toString() << ". Returning "
                  << ResultIt->Entry().toString() << ".");
      return &*ResultIt;
    } else {
      // We have a single match of the given address
      return &*ResultIt;
    }
  }

  void setEntryPoint(const MetaAddress &Address) {
    revng_assert(Address.isValid());

    if (not isExecutable(Address)) {
      report("set EntryPoint", Address);
      return;
    }

    Binary->EntryPoint() = Address;
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
    return ExecutableRanges.contains(Address);
  }

protected:
  ImportLogger importLogger(llvm::StringRef Path) {
    return ImportLogger(Binary, Logger, Path);
  }

  template<IsObjectFile T>
  std::optional<LDDTree>
  identifyDependencies(const T &ObjectFile, llvm::StringRef CanonicalPath);
};
