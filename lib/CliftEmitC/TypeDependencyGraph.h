#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/GenericGraph.h"

namespace mlir {

class ModuleOp;

namespace clift {

class DefinedType;

/// Represents a defined type in the dependency graph.
struct TypeDefinitionNode {

  const mlir::clift::DefinedType T;

  /// For each defined type we'll have nodes representing the type name or
  // the full type, depending on this enum.
  enum Kind {
    /// The TypeNode represents a declaration of a type name
    Declaration,
    /// The TypeNode represents the full definition of a type name
    Definition,
  } K;

  bool isDeclaration() const { return K == Declaration; }

  bool isDefinition() const { return K == Definition; }
};

using TypeDependencyNode = BidirectionalNode<TypeDefinitionNode>;

/// Represents the graph of dependencies among types
class TypeDependencyGraph : public GenericGraph<TypeDependencyNode> {
public:
  /// An internal factory class used to build a DependencyGraph.
  template<bool HelperMode>
  class Builder;

private:
  /// A pair of associated nodes that are respectively the declaration and the
  /// definition of the same type.
  struct AssociatedNodes {
    TypeDependencyNode *Declaration;
    TypeDependencyNode *Definition;
  };

  // Allow using `mlir::clift::DefinedType` as a map key.
  struct HandleComparator {
    auto operator()(mlir::clift::DefinedType LHS,
                    mlir::clift::DefinedType RHS) const {
      return LHS.getHandle() < RHS.getHandle();
    }
  };

  /// A map type that maps a type definition to a pair of nodes, representing
  //  respectively the declaration and the definition of such type definition.
  using TypeToNodesMap = std::map<mlir::clift::DefinedType, // formatting
                                  AssociatedNodes,
                                  HandleComparator>;

  /// Maps a type definition to its associated nodes, representing
  /// respectively its declaration and definition.
  TypeToNodesMap TypeToNodes;

public:
  /// Factory method to create a type dependency graph from an mlir module.
  static TypeDependencyGraph makeModelGraph(const mlir::ModuleOp &Module,
                                            uint64_t TargetPointerSize);

  /// Factory method to create a helper dependency graph from an mlir module.
  static TypeDependencyGraph makeHelperGraph(const mlir::ModuleOp &Module,
                                             uint64_t TargetPointerSize);

public:
  /// Helper debug method. It visualizes the graph, invoking xdot.
  void viewGraph() const debug_function;
};

std::string getNodeLabel(const TypeDependencyNode *N);

} // namespace clift
} // namespace mlir

template<>
struct llvm::DOTGraphTraits<mlir::clift::TypeDependencyGraph *>
  : public llvm::DefaultDOTGraphTraits {

  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  std::string getNodeLabel(const mlir::clift::TypeDependencyNode *N,
                           const mlir::clift::TypeDependencyGraph *G);
};
