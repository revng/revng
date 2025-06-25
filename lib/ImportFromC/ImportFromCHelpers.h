#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"
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
#include "revng/TypeNames/ModelCBuilder.h"

namespace {

struct NodeData {
  const model::TypeDefinition *T;
};
using Node = BidirectionalNode<NodeData>;
using Graph = GenericGraph<Node>;
using TypeToNodeMap = std::map<const model::TypeDefinition *, Node *>;

inline std::set<model::TypeDefinition::Key>
getIncomingTypesFor(const model::TypeDefinition &T,
                    const TupleTree<model::Binary> &Model,
                    const TypeToNodeMap &TypeToNode) {
  std::set<model::TypeDefinition::Key> Result;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N : depth_first_ext(TypeToNode.at(&T), Visited))
    ;

  for (auto &Type : Model->TypeDefinitions()) {
    if (Visited.contains(TypeToNode.at(Type.get())))
      Result.insert(Type->key());
  }

  return Result;
}

// Remember dependencies between types.
inline std::set<model::TypeDefinition::Key>
collectDependentTypes(const model::TypeDefinition &TheType,
                      const TupleTree<model::Binary> &Model) {
  std::set<model::TypeDefinition::Key> Result;

  Graph InverseTypeGraph;
  std::map<const model::TypeDefinition *, Node *> TypeToNode;

  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    TypeToNode[T.get()] = InverseTypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions())
    for (const model::Type *Edge : T->edges())
      if (auto *DependantType = Edge->skipToDefinition())
        TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));

  // Process types.
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    for (const model::Type *Edge : T->edges()) {
      if (auto *DependantType = Edge->skipToDefinition()) {
        // For non-typedefs, we can only keep the types that use the type
        // we are about to edit if there's a pointer in-between.
        if (!ptml::ModelCBuilder::isDeclarationTheSameAsDefinition(TheType)
            and Edge->isPointer())
          continue;

        // Current type depends on TheType.
        if (*DependantType == TheType) {
          Result.insert(T->key());

          // We need to skip all the types that can reach to this type we have
          // just ignored by doing DFS on the inverse graph.
          auto AllThatUseTypeT = getIncomingTypesFor(*T, Model, TypeToNode);
          llvm::copy(AllThatUseTypeT, std::inserter(Result, Result.begin()));
        }
      }
    }
  }

  // Skip the type itself.
  Result.insert(TheType.key());

  return Result;
}

} // namespace
