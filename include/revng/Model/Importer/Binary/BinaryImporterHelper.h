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

  /// Resolve an already-relocated raw code address to the MetaAddress the
  /// model uses for that function.
  ///
  /// \p Address must be pre-relocated by the caller (use
  /// `relocate(uint64_t).address()`). \p Architecture provides the code
  /// types to probe.
  ///
  /// `fromPC` builds the canonical MetaAddress (honoring per-arch
  /// conventions like ARM's Thumb LSB). The alternative code types are
  /// then probed at the same numeric address with the explicit
  /// MetaAddress constructor, which validates the per-type alignment.
  ///
  /// If `Binary->Functions()` already has an entry for one of the
  /// candidates, that exact MetaAddress is returned so the caller's model
  /// and whitelist lookups match. Otherwise the first valid candidate is
  /// returned so the caller can still insert a fresh entry at it.
  ///
  /// This function does not touch the model.
  MetaAddress
  matchFunctionEntry(uint64_t Address,
                     model::Architecture::Values Architecture) const {
    using namespace MetaAddressType;
    llvm::SmallVector<MetaAddress, 2> Candidates;

    MetaAddress Primary = MetaAddress::fromPC(Architecture, Address);
    if (Primary.isValid())
      Candidates.push_back(Primary);

    uint64_t BareAddress = Primary.isValid() ? Primary.address() : Address;
    for (auto CodeType : archCodeTypes(Architecture)) {
      if (Primary.isValid() and CodeType == Primary.type())
        continue;
      MetaAddress Alt(BareAddress, CodeType);
      if (Alt.isValid())
        Candidates.push_back(Alt);
    }

    if (Candidates.empty())
      return MetaAddress::invalid();

    if (not isExecutable(Candidates.front())) {
      report("match Function", Candidates.front());
      return MetaAddress::invalid();
    }

    for (const auto &Candidate : Candidates)
      if (Binary->Functions().find(Candidate) != Binary->Functions().end())
        return Candidate;
    return Candidates.front();
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
  void report(const char *Action, const MetaAddress &Address) const {
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
