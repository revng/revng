/// \file PromoteOriginalName.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

void recordCustomNamesInList(auto &Collection,
                             auto Unwrap,
                             std::set<std::string> &UsedNames) {
  for (auto &Entry2 : Collection) {
    auto *Entry = Unwrap(Entry2);
    if (not Entry->CustomName().empty())
      UsedNames.insert(Entry->CustomName().str().str());
  }
}

void promoteOriginalNamesInList(auto &Collection,
                                auto Unwrap,
                                std::set<std::string> &UsedNames) {
  // TODO: collapse uint8_t typedefs into the primitive type

  for (auto &Entry2 : Collection) {
    auto *Entry = Unwrap(Entry2);
    if (Entry->CustomName().empty() and not Entry->OriginalName().empty()) {
      // We have an OriginalName but not CustomName
      auto Name = Identifier::fromString(Entry->OriginalName());

      while (UsedNames.contains(Name.str().str()))
        Name += "_";

      // Assign name
      Entry->CustomName() = Name;

      // Record new name
      UsedNames.insert(Name.str().str());
    }
  }
}

void promoteOriginalNamesInList(auto &Collection, auto Unwrap) {
  std::set<std::string> UsedNames;
  recordCustomNamesInList(Collection, Unwrap, UsedNames);
  promoteOriginalNamesInList(Collection, Unwrap, UsedNames);
}

/// Promote OriginalNames to CustomNames
void model::promoteOriginalName(TupleTree<model::Binary> &Model) {
  auto AddressOf = [](auto &Entry) { return &Entry; };
  auto Unwrap = [](auto &UC) { return UC.get(); };

  // Collect all the already used CustomNames for symbols
  std::set<std::string> Symbols;
  recordCustomNamesInList(Model->Types(), Unwrap, Symbols);
  recordCustomNamesInList(Model->Functions(), AddressOf, Symbols);
  recordCustomNamesInList(Model->ImportedDynamicFunctions(),
                          AddressOf,
                          Symbols);
  for (auto &UP : Model->Types())
    if (auto *Enum = dyn_cast<EnumType>(UP.get()))
      recordCustomNamesInList(Enum->Entries(), AddressOf, Symbols);

  // Promote type names
  promoteOriginalNamesInList(Model->Types(), Unwrap, Symbols);

  // Promote function names
  promoteOriginalNamesInList(Model->Functions(), AddressOf, Symbols);

  // Promote dynamic function names
  promoteOriginalNamesInList(Model->ImportedDynamicFunctions(),
                             AddressOf,
                             Symbols);

  // Promote segment names
  promoteOriginalNamesInList(Model->Segments(), AddressOf, Symbols);

  for (auto &UP : Model->Types()) {
    model::Type *T = UP.get();

    if (auto *Struct = dyn_cast<StructType>(T)) {
      // Promote struct fields names (they have their own namespace)
      promoteOriginalNamesInList(Struct->Fields(), AddressOf);
    } else if (auto *Union = dyn_cast<UnionType>(T)) {
      // Promote union fields names (they have their own namespace)
      promoteOriginalNamesInList(Union->Fields(), AddressOf);
    } else if (auto *CFT = dyn_cast<CABIFunctionType>(T)) {
      // Promote argument names (they have their own namespace)
      promoteOriginalNamesInList(CFT->Arguments(), AddressOf);
    } else if (auto *Enum = dyn_cast<EnumType>(T)) {
      // Promote enum entries names (they are symbols)
      promoteOriginalNamesInList(Enum->Entries(), AddressOf, Symbols);
    }
  }
}
