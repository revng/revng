#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "mlir/IR/BuiltinOps.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Clift/CliftTypeInterfaces.h"

/// Represents a defined type in the dependency graph.
struct DefinedTypeNode {
  const clift::DefinedType T;
  const bool IsDefinition = true;

  static DefinedTypeNode definition(const clift::DefinedType T) {
    return DefinedTypeNode(T, true);
  }

  static DefinedTypeNode declaration(const clift::DefinedType T) {
    return DefinedTypeNode(T, false);
  }

  std::string label() const;
};

using TypeDependencyNode = BidirectionalNode<DefinedTypeNode>;

/// Represents the graph of dependencies among types
class TypeDependencyGraph : public GenericGraph<TypeDependencyNode> {
private:
  /// A pair of associated nodes that are respectively the declaration and the
  /// definition of the same type.
  struct AssociatedNodes {
    TypeDependencyNode *Declaration;
    TypeDependencyNode *Definition;
  };

  // Allow using `clift::DefinedType` as a map key.
  struct HandleComparator {
    auto operator()(clift::DefinedType LHS, clift::DefinedType RHS) const {
      return LHS.getHandle() < RHS.getHandle();
    }
  };

  /// A map type that maps a type definition to a pair of nodes, representing
  //  respectively the declaration and the definition of such type definition.
  using TypeToNodesMap = std::map<clift::DefinedType, // formatting
                                  AssociatedNodes,
                                  HandleComparator>;

  /// Maps a type definition to its associated nodes, representing
  /// respectively its declaration and definition.
  TypeToNodesMap TypeToNodes;

public:
  /// Factory method to create a type dependency graph from an mlir module.
  static TypeDependencyGraph makeModelGraph(mlir::ModuleOp Module);

  /// Factory method to create a helper dependency graph from an mlir module.
  static TypeDependencyGraph
  makeHelperGraph(llvm::ArrayRef<mlir::ModuleOp> Module);

public:
  /// Helper debug method. It visualizes the graph, invoking xdot.
  void viewGraph() const debug_function;

private:
  template<bool ModelMode>
  class Builder;
};

template<>
struct llvm::DOTGraphTraits<TypeDependencyGraph *>
  : public llvm::DefaultDOTGraphTraits {

  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  std::string getNodeLabel(const TypeDependencyNode *N,
                           const TypeDependencyGraph *G);
};
