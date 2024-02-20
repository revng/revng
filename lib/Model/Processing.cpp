/// \file Processing.cpp
/// A collection of helper functions to improve the quality of the model/make it
/// valid.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Processing.h"
#include "revng/Support/Debug.h"

using namespace llvm;

namespace model {

template<typename T>
void purgeFunctions(T &Functions,
                    const std::set<const model::TypeDefinition *> &ToDelete) {
  auto Begin = Functions.begin();
  for (auto It = Begin; It != Functions.end(); ++It) {
    if (not It->Prototype().isValid()
        or ToDelete.contains(It->Prototype().get())) {
      It->Prototype() = TupleTreeReference<model::TypeDefinition,
                                           model::Binary>();
    }
  }
}

using DefinitionPointerSet = std::set<const model::TypeDefinition *>;
unsigned dropTypesDependingOnDefinitions(TupleTree<model::Binary> &Model,
                                         const DefinitionPointerSet &Types) {
  // TODO: in case we reach a StructField or UnionField, we should drop the
  //       field and not proceed any further
  struct TypeNode {
    const model::TypeDefinition *T;
  };

  using Graph = GenericGraph<ForwardNode<TypeNode>>;

  Graph ReverseDependencyGraph;

  // Create nodes in reverse dependency graph
  std::map<const model::TypeDefinition *, ForwardNode<TypeNode> *> TypeToNode;
  for (model::UpcastableTypeDefinition &T : Model->TypeDefinitions())
    TypeToNode[T.get()] = ReverseDependencyGraph.addNode(TypeNode{ T.get() });

  // Register edges
  for (model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    // Ignore dependencies of types we need to drop
    if (Types.contains(T.get()))
      continue;

    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));
    }
  }

  // Prepare for deletion all the nodes reachable from Types
  std::set<const model::TypeDefinition *> ToDelete;
  for (const model::TypeDefinition *Type : Types) {
    for (const auto *Node : depth_first(TypeToNode.at(Type))) {
      ToDelete.insert(Node->T);
    }
  }

  // Purge both dynamic and local functions depending on Types
  purgeFunctions(Model->ImportedDynamicFunctions(), ToDelete);
  purgeFunctions(Model->Functions(), ToDelete);

  // Purge types depending on unresolved Types
  for (auto It = Model->TypeDefinitions().begin();
       It != Model->TypeDefinitions().end();) {
    if (ToDelete.contains(It->get()))
      It = Model->TypeDefinitions().erase(It);
    else
      ++It;
  }

  return ToDelete.size();
}

} // namespace model
