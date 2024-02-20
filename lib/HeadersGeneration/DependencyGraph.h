#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/TypeDefinition.h"

/// Represents a model::TypeDefinition in the DependencyGraph
struct TypeNode {

  /// A pointer to the associated model::TypeDefinition
  const model::TypeDefinition *T;

  /// For each model::TypeDefinition we'll have nodes representing the type name
  /// or the full type, depending on this enum.
  enum Kind {
    Declaration,
    Definition
  } K;
};

using TypeDependencyNode = BidirectionalNode<TypeNode>;
using TypeKindPair = std::pair<const model::TypeDefinition *, TypeNode::Kind>;
using TypeToDependencyNodeMap = std::map<TypeKindPair, TypeDependencyNode *>;
using TypeVector = TrackingSortedVector<model::UpcastableTypeDefinition>;

/// Represents the graph of dependencies among types
struct DependencyGraph : public GenericGraph<TypeDependencyNode> {

  void addNode(const model::TypeDefinition *T);

  const TypeToDependencyNodeMap &TypeNodes() const { return TypeToNode; }

private:
  TypeToDependencyNodeMap TypeToNode;
};

std::string getNodeLabel(const TypeDependencyNode *N);

template<>
struct llvm::DOTGraphTraits<DependencyGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  std::string getNodeLabel(const TypeDependencyNode *N,
                           const DependencyGraph *G);
};

DependencyGraph buildDependencyGraph(const TypeVector &Types);
