#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Type.h"

/// Represents a model::Type in the DependencyGraph
struct TypeNode {

  /// A pointer to the associated model::Type
  const model::Type *T;

  /// For each model::Type we'll have nodes representing the type name or
  /// the full type, depending on this enum.
  enum Kind {
    Declaration,
    Definition
  } K;
};

using TypeDependencyNode = BidirectionalNode<TypeNode>;
using TypeKindPair = std::pair<const model::Type *, TypeNode::Kind>;
using TypeToDependencyNodeMap = std::map<TypeKindPair, TypeDependencyNode *>;
using TypeVector = TrackingSortedVector<UpcastablePointer<model::Type>>;

/// Represents the graph of dependencies among types
struct DependencyGraph : public GenericGraph<TypeDependencyNode> {

  void addNode(const model::Type *T);

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
