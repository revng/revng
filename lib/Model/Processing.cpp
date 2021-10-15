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

  auto RegisterDependency = [&](UpcastablePointer<model::Type> &T,
                                const model::QualifiedType &QT) {
    auto *DependantType = QT.UnqualifiedType.get();
    TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));
  };

  // Populate the graph
  for (UpcastablePointer<model::Type> &T : Model->Types) {

    // Ignore dependencies of
    if (Types.count(T.get()) != 0)
      continue;

    if (auto *Primitive = dyn_cast<model::PrimitiveType>(T.get())) {
      // Nothing to do here
    } else if (auto *Struct = dyn_cast<model::StructType>(T.get())) {
      for (const model::StructField &Field : Struct->Fields)
        RegisterDependency(T, Field.Type);
    } else if (auto *Union = dyn_cast<model::UnionType>(T.get())) {
      for (const model::UnionField &Field : Union->Fields)
        RegisterDependency(T, Field.Type);
    } else if (auto *Enum = dyn_cast<model::EnumType>(T.get())) {
      RegisterDependency(T, { Enum->UnderlyingType });
    } else if (auto *Typedef = dyn_cast<model::TypedefType>(T.get())) {
      RegisterDependency(T, Typedef->UnderlyingType);
    } else if (auto *RFT = dyn_cast<model::RawFunctionType>(T.get())) {
      for (const model::NamedTypedRegister &Argument : RFT->Arguments)
        RegisterDependency(T, Argument.Type);
      for (const model::TypedRegister &RV : RFT->ReturnValues)
        RegisterDependency(T, RV.Type);
    } else if (auto *CAFT = dyn_cast<model::CABIFunctionType>(T.get())) {
      for (const model::Argument &Argument : CAFT->Arguments)
        RegisterDependency(T, Argument.Type);
      RegisterDependency(T, CAFT->ReturnType);
    } else {
      revng_abort();
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

void deduplicateNames(TupleTree<model::Binary> &Model) {
  // TODO: collapse uint8_t typedefs into the primitive type

  std::set<std::string> UsedNames;

  for (auto &Type : Model->Types) {
    model::Type *T = Type.get();
    if (isa<model::PrimitiveType>(T)) {
      UsedNames.insert(T->name().str().str());
    }
  }

  for (auto &Type : Model->Types) {
    model::Type *T = Type.get();
    if (isa<model::PrimitiveType>(T))
      continue;

    std::string Name = T->name().str().str();
    while (UsedNames.count(Name) != 0) {
      Name += "_";
    }

    // Rename
    upcast(T, [&Name](auto &Upcasted) {
      using UpcastedType = std::remove_cvref_t<decltype(Upcasted)>;
      if constexpr (not std::is_same_v<model::PrimitiveType, UpcastedType>) {
        Upcasted.CustomName = Name;
      } else {
        revng_abort();
      }
    });

    // Record new name
    UsedNames.insert(Name);
  }
}

void deduplicateEquivalentTypes(TupleTree<model::Binary> &Model) {
  // TODO: implement
  (void) Model;

  // Create strong equivalence classes

  // Create weak (candidate) equivalence classes based on the same name and
  // local equivalence

  // Create a bidirectional graph of the non-local parts (including pointers)

  // Mark nodes with no (or only pointer) predecessors as entry points

  // Do a post order visit starting from entry points

  // For the current node, consider all the weakly equivalent node pairs and
  // start comparing

  // Create a bidirectional map associating left and right nodes

  // Initialize the map with the two considered nodes

  // Do a dfs

  // Zip out edges of the node pair: consider the destinations.
  //
  // * If any of them is in the associating map, the other needs to match.
  //   If it doesn't, the two nodes are not equivalent.
  // * Otherwise, are they strongly/weakly equivalent? If so, insert them in
  //   the associating map and proceed.
  // * Otherwise, the nodes are not equivalent.
}

} // namespace model
