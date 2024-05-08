/// \file PromoteOriginalName.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <set>
#include <string>

#include "revng/Model/Pass/PromoteOriginalName.h"
#include "revng/Model/Pass/RegisterModelPass.h"

using namespace llvm;
using namespace model;

static RegisterModelPass R("promote-original-name",
                           "Promote OriginalName fields to CustomName ensuring "
                           "the validity of the model is preserved",
                           model::promoteOriginalName);

class SymbolPromoter {
private:
  std::set<Identifier> GlobalSymbols;
  std::set<Identifier> TakenLocalSymbols;

public:
  void dump() const debug_function {
    for (const Identifier &ID : GlobalSymbols)
      dbg << ID.str().str() << "\n";
  }

public:
  void recordGlobalSymbols(auto &Collection, auto Unwrap) {
    for (auto &Entry2 : Collection) {
      auto *Entry = Unwrap(Entry2);
      if (not Entry->CustomName().empty()) {
        GlobalSymbols.insert(Entry->CustomName());
      }
    }
  }

  void recordLocalSymbols(auto &Collection, auto Unwrap) {
    for (auto &Entry2 : Collection) {
      auto *Entry = Unwrap(Entry2);
      if (not Entry->CustomName().empty()) {
        TakenLocalSymbols.insert(Entry->CustomName());
      }
    }
  }

  void promoteGlobalSymbols(auto &Collection, auto Unwrap) {
    promoteSymbolsImpl(Collection, Unwrap, GlobalSymbols, TakenLocalSymbols);
  }

  void promoteLocalSymbols(auto &Collection, auto Unwrap) {
    std::set<Identifier> LocalBucket;
    promoteSymbolsImpl(Collection, Unwrap, LocalBucket, GlobalSymbols);
  }

private:
  void promoteSymbolsImpl(auto &Collection,
                          auto Unwrap,
                          std::set<Identifier> &Namespace,
                          const std::set<Identifier> &Taken) {
    // TODO: collapse uint8_t typedefs into the primitive type
    for (auto &Wrapped : Collection) {
      auto *Entry = Unwrap(Wrapped);
      if (Entry->CustomName().empty() and not Entry->OriginalName().empty()) {
        // We have an OriginalName but not CustomName
        auto Name = Identifier::fromString(Entry->OriginalName());

        while (Taken.contains(Name) or Namespace.contains(Name))
          Name += "_";

        // Assign name
        Entry->CustomName() = Name;

        // Record new name as taken in the current namespace
        auto [_, Inserted] = Namespace.insert(Name);
        revng_assert(Inserted);
      }
    }
  }
};

/// Promote OriginalNames to CustomNames
void model::promoteOriginalName(TupleTree<model::Binary> &Model) {
  auto AddressOf = [](auto &Entry) { return &Entry; };
  auto Unwrap = [](auto &UC) { return UC.get(); };

  SymbolPromoter Promoter;

  // Reserve symbols we can't use both for local symbols and global symbols
  Promoter.recordGlobalSymbols(Model->Functions(), AddressOf);
  Promoter.recordGlobalSymbols(Model->ImportedDynamicFunctions(), AddressOf);
  Promoter.recordGlobalSymbols(Model->Types(), Unwrap);
  for (auto &UP : Model->Types())
    if (auto *Enum = dyn_cast<EnumType>(UP.get()))
      Promoter.recordGlobalSymbols(Enum->Entries(), AddressOf);

  Promoter.recordGlobalSymbols(Model->Segments(), AddressOf);

  // Reserve symbols we can't use for global symbols
  for (auto &UP : Model->Types()) {
    model::Type *T = UP.get();

    if (auto *Struct = dyn_cast<StructType>(T)) {
      Promoter.recordLocalSymbols(Struct->Fields(), AddressOf);
    } else if (auto *Union = dyn_cast<UnionType>(T)) {
      Promoter.recordLocalSymbols(Union->Fields(), AddressOf);
    } else if (auto *CFT = dyn_cast<CABIFunctionType>(T)) {
      Promoter.recordLocalSymbols(CFT->Arguments(), AddressOf);
    } else if (auto *RFT = dyn_cast<RawFunctionType>(T)) {
      Promoter.recordLocalSymbols(RFT->Arguments(), AddressOf);
    }
  }

  // Promote global symbols
  Promoter.promoteGlobalSymbols(Model->Functions(), AddressOf);
  Promoter.promoteGlobalSymbols(Model->ImportedDynamicFunctions(), AddressOf);
  Promoter.promoteGlobalSymbols(Model->Types(), Unwrap);
  for (auto &UP : Model->Types())
    if (auto *Enum = dyn_cast<EnumType>(UP.get()))
      Promoter.promoteGlobalSymbols(Enum->Entries(), AddressOf);

  Promoter.promoteGlobalSymbols(Model->Segments(), AddressOf);

  // Promote local symbols
  for (auto &UP : Model->Types()) {
    model::Type *T = UP.get();

    if (auto *Struct = dyn_cast<StructType>(T)) {
      Promoter.promoteLocalSymbols(Struct->Fields(), AddressOf);
    } else if (auto *Union = dyn_cast<UnionType>(T)) {
      Promoter.promoteLocalSymbols(Union->Fields(), AddressOf);
    } else if (auto *CFT = dyn_cast<CABIFunctionType>(T)) {
      Promoter.promoteLocalSymbols(CFT->Arguments(), AddressOf);
    } else if (auto *RFT = dyn_cast<RawFunctionType>(T)) {
      Promoter.promoteLocalSymbols(RFT->Arguments(), AddressOf);
    }
  }
}
