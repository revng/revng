#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Option.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"

namespace {

struct NodeData {
  model::Type *T;
};
using Node = BidirectionalNode<NodeData>;
using Graph = GenericGraph<Node>;

inline llvm::SmallPtrSet<const model::Type *, 2>
getIncomingTypesFor(const model::Type *T,
                    const TupleTree<model::Binary> &Model,
                    const std::map<const model::Type *, Node *> &TypeToNode) {
  llvm::SmallPtrSet<const model::Type *, 2> Result;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N : depth_first_ext(TypeToNode.at(T), Visited))
    ;

  for (auto &Type : Model->Types()) {
    if (Visited.contains(TypeToNode.at(Type.get())))
      Result.insert(Type.get());
  }

  return Result;
}

// Remember dependencies between types.
inline llvm::SmallPtrSet<const model::Type *, 2>
populateDependencies(const model::Type *TheType,
                     const TupleTree<model::Binary> &Model) {
  llvm::SmallPtrSet<const model::Type *, 2> Result;

  Graph InverseTypeGraph;
  std::map<const model::Type *, Node *> TypeToNode;

  for (const UpcastablePointer<model::Type> &T : Model->Types()) {
    TypeToNode[T.get()] = InverseTypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges
  for (const UpcastablePointer<model::Type> &T : Model->Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));
    }
  }

  // Process types.
  for (const UpcastablePointer<model::Type> &T : Model->Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();

      // For types other than TypeDefs, we can keep the types that use the type
      // we are about to edit iff the type is being used via pointer.
      if ((not llvm::isa<model::TypedefType>(TheType)
           and not llvm::isa<model::RawFunctionType>(TheType)
           and not llvm::isa<model::CABIFunctionType>(TheType))
          and QT.isPointer())
        continue;

      // This type depends on TheType.
      if (DependantType == TheType) {
        Result.insert(T.get());

        // We need to skip all the types that can reach to this type we have
        // just ignored by doing DFS on the inverse graph.
        auto AllThatUseTypeT = getIncomingTypesFor(T.get(), Model, TypeToNode);
        llvm::copy(AllThatUseTypeT, std::inserter(Result, Result.begin()));
      }
    }
  }

  // Skip the type itself.
  Result.insert(TheType);

  return Result;
}
} // namespace
