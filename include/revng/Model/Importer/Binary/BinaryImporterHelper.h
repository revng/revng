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

  /// \return the MetaAddress of a function at the relocated address \p Address,
  /// possibly already existing in the model.
  MetaAddress
  matchFunctionEntry(uint64_t Address,
                     model::Architecture::Values Architecture) const {
    using namespace MetaAddressType;
    llvm::SmallVector<MetaAddress, 2> Candidates;

    auto Primary = MetaAddress::fromPC(Architecture, Address);
    if (Primary.isValid())
      Candidates.push_back(Primary);

    uint64_t BareAddress = Primary.isValid() ? Primary.address() : Address;
    for (auto CodeType : archCodeTypes(Architecture)) {

      if (Primary.isValid() and CodeType == Primary.type())
        continue;

      MetaAddress Alternative(BareAddress, CodeType);
      if (Alternative.isValid())
        Candidates.push_back(Alternative);
    }

    if (Candidates.empty())
      return MetaAddress::invalid();

    for (const auto &Candidate : Candidates)
      if (Binary->Functions().contains(Candidate))
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
