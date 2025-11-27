//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "revng/ADT/ScopedExchange.h"
#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

static Logger Log{ "clift-type-dependency-graph" };

std::string mlir::clift::DefinedTypeNode::label() const {
  return "[" + T.getHandle().str() + " "
         + (IsDefinition ? "definition" : "declaration") + "]";
}

using DepNode = mlir::clift::TypeDependencyNode;
using DepGraph = mlir::clift::TypeDependencyGraph;
std::string llvm::DOTGraphTraits<DepGraph *>::getNodeLabel(const DepNode *N,
                                                           const DepGraph *G) {
  return N->label();
}

template<typename VisitorT>
using ModuleVisitor = mlir::clift::ModuleVisitor<VisitorT>;

template<bool ModelMode>
using Builder = mlir::clift::TypeDependencyGraph::Builder<ModelMode>;

/// This is a private graph builder.
///
/// \tparam ModelMode This parameter decides whether the graph is built
/// exclusively from model types (so starting from a function, a segment or
/// a `clift.types` attribute) OR from everything else (in practice, just
/// helpers).
template<bool ModelMode>
class mlir::clift::TypeDependencyGraph::Builder
  : public ModuleVisitor<Builder<ModelMode>> {
  /// A pointer to the dependency graph being constructed and initialized.
  mlir::clift::TypeDependencyGraph *Graph = nullptr;

  using Base = ModuleVisitor<Builder<ModelMode>>;

public:
  Builder(mlir::clift::TypeDependencyGraph &Graph) : Graph(&Graph) {}

  /// Create and initialize a dependency graph.
  static mlir::clift::TypeDependencyGraph
  make(const std::vector<mlir::ModuleOp> &Modules) {
    // Import all the nodes
    mlir::clift::TypeDependencyGraph Result;
    for (mlir::ModuleOp Module : Modules) {
      mlir::LogicalResult MaybeError = Base::visit(Module, Result);
      revng_assert(MaybeError.succeeded());
    }

    // And all the edges
    for (auto [Type, AssociatedNodes] : Result.TypeToNodes)
      Builder(Result).addDependencies(Type, AssociatedNodes);

    if (Log.isEnabled())
      llvm::ViewGraph(&Result, "type-deps.dot");

    return Result;
  }

  /// The visitor implementation for the type system traversal
  mlir::LogicalResult visitType(mlir::Type Type);

private:
  /// Add a declaration node and a definition node to Graph for \p Type.
  void addNodes(mlir::clift::DefinedType Type) const;

  /// Add all the necessary dependency edges to Graph for the nodes that
  /// represent the declaration and definition of \p Type.
  void addDependencies(mlir::clift::DefinedType Type,
                       AssociatedNodes Nodes) const;

  /// Add all the necessary dependency edges from the \p Dependent to the
  /// nodes associated with the \p DependedOn type.
  void addDependenciesFrom(const AssociatedNodes Dependent,
                           mlir::clift::ValueType DependedOn) const;
};

static void addAndLogSuccessor(mlir::clift::TypeDependencyNode *From,
                               mlir::clift::TypeDependencyNode *To) {
  revng_assert(From);
  revng_assert(To);
  revng_log(Log, "Added edge: " << From->label() << " --> " << To->label());
  From->addSuccessor(To);
}

template<bool Mode>
void Builder<Mode>::addNodes(mlir::clift::DefinedType Type) const {
  if (Graph->TypeToNodes.contains(Type)) {
    // There is already a node for this type.
    return;
  }

  using DefinedTypeNode = mlir::clift::DefinedTypeNode;
  auto *DeclNode = Graph->addNode(DefinedTypeNode::declaration(Type));

  revng_log(Log, "Added node: " << DeclNode->label());

  mlir::clift::TypeDependencyNode *DefNode = nullptr;
  if (mlir::clift::isSeparateDeclarationAllowed(Type)) {
    DefNode = Graph->addNode(DefinedTypeNode::definition(Type));

    revng_log(Log, "Added node: " << DefNode->label());

    // The definition always depends on the declaration.
    // This is not strictly necessary (e.g. when definition and declaration are
    // the same, or when printing a the body of a struct without having forward
    // declared it) but it doesn't introduce cycles and it enables the algorithm
    // that decides on the ordering on the declarations and definitions to make
    // more assumptions about definitions being emitted before declarations.
    addAndLogSuccessor(DefNode, DeclNode);
  }

  revng_assert(not Graph->TypeToNodes.contains(Type));
  Graph->TypeToNodes[Type] = AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

struct TypeSpecifierResult {
  // `nullopt` means that there is no target type, as in, this is either
  // a primitive or a primitive-like structure (currently only foreign pointers,
  // but we might need others in the future). And those are assumed to always
  // be available in a different header.
  mlir::clift::DefinedType Definition = nullptr;
  bool FoundPointer = false;
  bool LastArray = false;
};

static TypeSpecifierResult unwrapType(mlir::clift::ValueType T) {
  TypeSpecifierResult Result;

  while (true) {

    if (auto Pointer = mlir::dyn_cast<mlir::clift::PointerType>(T)) {
      Result.FoundPointer = true;
      Result.LastArray = false;

      T = Pointer.getPointeeType();

    } else if (auto Array = mlir::dyn_cast<mlir::clift::ArrayType>(T)) {
      Result.LastArray = true;

      T = Array.getElementType();

    } else if (auto Defined = mlir::dyn_cast<mlir::clift::DefinedType>(T)) {
      Result.Definition = Defined;

      return Result;

    } else if (auto Primitive = mlir::dyn_cast<mlir::clift::PrimitiveType>(T)) {
      Result.Definition = nullptr;

      return Result;

    } else {
      revng_abort("Unknown value type.");
    }
  }

  revng_abort("Unreachable.");
}

template<bool Mode>
void Builder<Mode>::addDependenciesFrom(const AssociatedNodes Dependent,
                                        mlir::clift::ValueType DOn) const {
  const auto &[DefinitionDependedOn, FoundPointer, LastArray] = unwrapType(DOn);

  // If the definition this depends on is not a type definition, we're done,
  // because it cannot be a _real_ dependency. It's either a primitive, an array
  // of primitives, a pointer to a primitive or a primitive-like foreign
  // pointer.
  if (not DefinitionDependedOn)
    return;

  const auto &[DeclarationNode, DefinitionNode] = Dependent;
  revng_assert(DeclarationNode);

  mlir::clift::DefinedType DependentType = DeclarationNode->T;

  bool ForwardDeclaration = isSeparateDeclarationAllowed(DependentType);
  revng_assert(ForwardDeclaration == static_cast<bool>(DefinitionNode));
  TypeDependencyNode *DependentNode = ForwardDeclaration ? DefinitionNode :
                                                           DeclarationNode;

  auto NodesDependedOn = Graph->TypeToNodes.at(DefinitionDependedOn);
  revng_assert(NodesDependedOn.Declaration);

  // If `LastArray` is true, the node depended on is always the node
  // representing the full definition of the type.
  //
  // Note, that it would still be a `Declaration` node in cases where there is
  // no separate Definition (e.g. for typedefs (including function type ones)).
  if (LastArray) {
    TypeDependencyNode *NodeDependedOn = NodesDependedOn.Definition ?
                                           NodesDependedOn.Definition :
                                           NodesDependedOn.Declaration;
    revng_assert(NodeDependedOn);
    addAndLogSuccessor(DependentNode, NodeDependedOn);
    return;
  }

  // If `FoundPointer` is true, and `LastArray` is false, the node depended on
  // is always the `Declaration` node, because we don't need the full
  // definition.
  if (FoundPointer) {
    addAndLogSuccessor(DependentNode, NodesDependedOn.Declaration);
    return;
  }

  // Otherwise we fall back in the baseline case.

  // The `DependentNode` always depends on the `Declaration` node.
  addAndLogSuccessor(DependentNode, NodesDependedOn.Declaration);

  // If both `Dependent` and `DefinitionDependedOn` have a separate forward
  // declaration we add a dependency from `DependentNode` to the `Definition`
  // of the `NodesDependedOn`.
  if (ForwardDeclaration
      and mlir::clift::isSeparateDeclarationAllowed(DefinitionDependedOn)) {
    addAndLogSuccessor(DependentNode, NodesDependedOn.Definition);
  }

  // Finally, if the `DependentDefinition` has a forward declaration, it also
  // means that it has a separate definition from the forward declaration.
  // In that case, if `DefinitionDependedOn` is a typedef, we also have to look
  // across all those typedefs and ensure the full definition of the dependent
  // also depends on the full definition of the depended-on, across typedefs.
  if (ForwardDeclaration
      and mlir::isa<mlir::clift::TypedefType>(DefinitionDependedOn)) {
    auto Underlying = dealias(DefinitionDependedOn,
                              /* IgnoreQualifiers= */ true);
    if (auto Defined = mlir::dyn_cast<mlir::clift::DefinedType>(Underlying)) {
      if (mlir::clift::isSeparateDeclarationAllowed(Defined)) {
        auto TransitivelyDependedOn = Graph->TypeToNodes.at(Defined);
        revng_assert(TransitivelyDependedOn.Definition);
        addAndLogSuccessor(DependentNode, TransitivelyDependedOn.Definition);
      }
    }
  }
}

template<bool Mode>
void Builder<Mode>::addDependencies(mlir::clift::DefinedType Type,
                                    AssociatedNodes Nodes) const {
  if (auto StructOrUnion = mlir::dyn_cast<mlir::clift::ClassType>(Type)) {
    for (auto Field : StructOrUnion.getFields())
      addDependenciesFrom(Nodes, Field.getType());

  } else if (auto Enum = mlir::dyn_cast<mlir::clift::EnumType>(Type)) {
    addDependenciesFrom(Nodes, Enum.getUnderlyingType());

  } else if (auto Typedef = mlir::dyn_cast<mlir::clift::TypedefType>(Type)) {
    addDependenciesFrom(Nodes, Typedef.getUnderlyingType());

  } else if (auto Function = mlir::dyn_cast<mlir::clift::FunctionType>(Type)) {
    addDependenciesFrom(Nodes, Function.getReturnType());
    for (auto Argument : Function.getArgumentTypes())
      addDependenciesFrom(Nodes, Argument);

  } else {
    Type.dump();
    revng_abort("Unsupported dependency type");
  }
}

template<bool ModelMode>
mlir::LogicalResult Builder<ModelMode>::visitType(mlir::Type T) {
  if (auto Type = mlir::dyn_cast<mlir::clift::DefinedType>(T)) {
    namespace rr = revng::ranks;
    bool ShouldAdd = pipeline::genericLocationFromString(Type.getHandle(),
                                                         rr::Binary,
                                                         rr::TypeDefinition,
                                                         rr::PrimitiveType,
                                                         rr::ArtificialStruct)
                       .has_value();
    if constexpr (not ModelMode)
      ShouldAdd = !ShouldAdd;

    // Never add helper prototypes, as we don't want to print those.
    if (pipeline::locationFromString(rr::HelperFunction, Type.getHandle()))
      ShouldAdd = false;

    if (ShouldAdd)
      if (auto TD = mlir::dyn_cast<mlir::clift::DefinedType>(Type))
        addNodes(TD);

    if (not ShouldAdd)
      revng_log(Log, "Skipping: " << Type.getHandle().str() << '\n');
  }

  return mlir::success();
}

mlir::clift::TypeDependencyGraph
mlir::clift::TypeDependencyGraph::makeModelGraph(mlir::ModuleOp Module) {
  return Builder<true>::make({ Module });
}

using TDG = mlir::clift::TypeDependencyGraph;
TDG TDG::makeHelperGraph(const std::vector<mlir::ModuleOp> &Modules) {
  return Builder<false>::make(Modules);
}

void mlir::clift::TypeDependencyGraph::viewGraph() const {
  llvm::ViewGraph(this, "type-dependency-graph.dot");
}
