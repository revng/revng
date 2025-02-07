#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

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
    /// The TypeNode represents a declaration of a type name
    Declaration,
    /// The TypeNode represents the full definition of a type name
    Definition,
    /// The TypeNode represents the forward declaration of an artificial struct
    /// wrapper for the type referred to by T
    ArtificialWrapperDeclaration,
    /// The TypeNode represents the full definition of an artificial struct
    /// wrapper for the type referred to by T
    ArtificialWrapperDefinition,
  } K;

  bool isDeclaration() const {
    return K == Declaration or K == ArtificialWrapperDeclaration;
  }

  bool isDefinition() const {
    return K == Definition or K == ArtificialWrapperDefinition;
  }

  bool isArtificial() const {
    return K == ArtificialWrapperDefinition
           or K == ArtificialWrapperDeclaration;
  }
};

using TypeDependencyNode = BidirectionalNode<TypeNode>;
using TypeVector = TrackingSortedVector<model::UpcastableTypeDefinition>;

/// Represents the graph of dependencies among types
class DependencyGraph : public GenericGraph<TypeDependencyNode> {
private:
  /// A pair of associated nodes that are respectively the declaration and the
  /// definition of the same type.
  struct AssociatedNodes {
    TypeDependencyNode *Declaration;
    TypeDependencyNode *Definition;
  };

  /// A internal factory class used to build a DependencyGraph
  class Builder;

  /// A map type that maps a model::TypeDefinition to a pair of
  /// TypeDependencyNodes, representing respectively the declaration and the
  /// definition of such TypeDefinition.
  using TypeToNodesMap = std::unordered_map<const model::TypeDefinition *,
                                            AssociatedNodes>;

  /// Maps a model::TypeDefinition * to an AssociatedNodes, representing
  /// respectively the declaration and the definition of the TypeDefinition.
  TypeToNodesMap TypeToNodes;

public:
  /// Factory method to create a DependencyGraph from a TypeVector.
  static DependencyGraph make(const TypeVector &TV);

public:
  /// Get the declaration node associated to \p Definition.
  /// Asserts if \p Definition is not a definition node.
  const TypeDependencyNode *
  getDeclaration(const model::TypeDefinition *TD) const {
    return TypeToNodes.at(TD).Declaration;
  }

  /// Get the definition node associated to \p Declaration.
  /// Asserts if \p Declaration is not a declaration node.
  const TypeDependencyNode *
  getDefinition(const model::TypeDefinition *TD) const {
    return TypeToNodes.at(TD).Definition;
  }
};

std::string getNodeLabel(const TypeDependencyNode *N);

template<>
struct llvm::DOTGraphTraits<DependencyGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  std::string getNodeLabel(const TypeDependencyNode *N,
                           const DependencyGraph *G);
};
