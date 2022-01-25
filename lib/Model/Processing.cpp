/// \file Processing.cpp
/// \brief A collection of helper functions to improve the quality of the
///        model/make it valid

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Processing.h"
#include "revng/Support/Debug.h"

#include "TypesDeduplication.h"

using namespace llvm;

namespace model {

unsigned dropTypesDependingOnTypes(TupleTree<model::Binary> &Model,
                                   const std::set<const model::Type *> &Types) {
  struct TypeNode {
    const model::Type *T;
  };

  using Graph = GenericGraph<ForwardNode<TypeNode>>;

  Graph ReverseDependencyGraph;

  // Create nodes in reverse dependency graph
  std::map<const model::Type *, ForwardNode<TypeNode> *> TypeToNode;
  for (UpcastablePointer<model::Type> &T : Model->Types)
    TypeToNode[T.get()] = ReverseDependencyGraph.addNode(TypeNode{ T.get() });

  // Register edges
  for (UpcastablePointer<model::Type> &T : Model->Types) {
    // Ignore dependencies of types we need to drop
    if (Types.count(T.get()) != 0)
      continue;

    for (model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType.get();
      TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));
    }
  }

  // Prepare for deletion all the nodes reachable from Types
  std::set<const model::Type *> ToDelete;
  for (const model::Type *Type : Types) {
    for (const auto *Node : depth_first(TypeToNode.at(Type))) {
      ToDelete.insert(Node->T);
    }
  }

  // Purge dynamic functions depending on Types
  auto Begin = Model->ImportedDynamicFunctions.begin();
  for (auto It = Begin; It != Model->ImportedDynamicFunctions.end(); /**/) {
    if (ToDelete.count(It->Prototype.get()) == 0) {
      ++It;
    } else {
      It = Model->ImportedDynamicFunctions.erase(It);
    }
  }

  // Purge types depending on unresolved Types
  for (auto It = Model->Types.begin(); It != Model->Types.end();) {
    if (ToDelete.count(It->get()) != 0)
      It = Model->Types.erase(It);
    else
      ++It;
  }

  return ToDelete.size();
}

void recordCustomNamesInList(auto &Collection,
                             auto Unwrap,
                             std::set<std::string> &UsedNames) {
  for (auto &Entry2 : Collection) {
    auto *Entry = Unwrap(Entry2);
    if (not Entry->CustomName.empty())
      UsedNames.insert(Entry->CustomName.str().str());
  }
}

void promoteOriginalNamesInList(auto &Collection,
                                auto Unwrap,
                                std::set<std::string> &UsedNames) {
  // TODO: collapse uint8_t typedefs into the primitive type

  for (auto &Entry2 : Collection) {
    auto *Entry = Unwrap(Entry2);
    if (Entry->CustomName.empty() and not Entry->OriginalName.empty()) {
      // We have an OriginalName but not CustomName
      auto Name = Identifier::fromString(Entry->OriginalName);

      while (UsedNames.count(Name.str().str()) != 0)
        Name += "_";

      // Assign name
      Entry->CustomName = Name;

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
void promoteOriginalName(TupleTree<model::Binary> &Model) {
  auto AddressOf = [](auto &Entry) { return &Entry; };
  auto Unwrap = [](auto &UC) { return UC.get(); };

  // Collect all the already used CustomNames for symbols
  std::set<std::string> Symbols;
  recordCustomNamesInList(Model->Types, Unwrap, Symbols);
  recordCustomNamesInList(Model->Functions, AddressOf, Symbols);
  recordCustomNamesInList(Model->ImportedDynamicFunctions, AddressOf, Symbols);
  for (auto &UP : Model->Types)
    if (auto *Enum = dyn_cast<EnumType>(UP.get()))
      recordCustomNamesInList(Enum->Entries, AddressOf, Symbols);

  // Promote type names
  promoteOriginalNamesInList(Model->Types, Unwrap, Symbols);

  // Promote function names
  promoteOriginalNamesInList(Model->Functions, AddressOf, Symbols);

  // Promote dynamic function names
  promoteOriginalNamesInList(Model->ImportedDynamicFunctions,
                             AddressOf,
                             Symbols);

  for (auto &UP : Model->Types) {
    model::Type *T = UP.get();

    if (auto *Struct = dyn_cast<StructType>(T)) {
      // Promote struct fields names (they have their own namespace)
      promoteOriginalNamesInList(Struct->Fields, AddressOf);
    } else if (auto *Union = dyn_cast<UnionType>(T)) {
      // Promote union fields names (they have their own namespace)
      promoteOriginalNamesInList(Union->Fields, AddressOf);
    } else if (auto *CFT = dyn_cast<CABIFunctionType>(T)) {
      // Promote argument names (they have their own namespace)
      promoteOriginalNamesInList(CFT->Arguments, AddressOf);
    } else if (auto *Enum = dyn_cast<EnumType>(T)) {
      // Promote enum entries names (they are symbols)
      promoteOriginalNamesInList(Enum->Entries, AddressOf, Symbols);
    }
  }
}

void deduplicateEquivalentTypes(TupleTree<model::Binary> &Model) {
  deduplicateEquivalentTypesImpl(Model);
}

} // namespace model
