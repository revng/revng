//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/ScopedExchange.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "TypeDependencyGraph.h"
#include "TypeStack.h"

static Logger<> Log{ "clift-type-dependency-graph" };

static llvm::StringRef toString(mlir::clift::TypeDefinitionNode::Kind K) {
  switch (K) {
  case mlir::clift::TypeDefinitionNode::Kind::Declaration:
    return "declaration";
  case mlir::clift::TypeDefinitionNode::Kind::Definition:
    return "definition";
  }
  return "Invalid";
}

std::string
mlir::clift::getNodeLabel(const mlir::clift::TypeDependencyNode *N) {
  return "[" + N->T.getHandle().str() + " " + toString(N->K).str() + "]";
}

using DepNode = mlir::clift::TypeDependencyNode;
using DepGraph = mlir::clift::TypeDependencyGraph;
std::string llvm::DOTGraphTraits<DepGraph *>::getNodeLabel(const DepNode *N,
                                                           const DepGraph *G) {
  return mlir::clift::getNodeLabel(N);
}

template<typename VisitorT>
using ModuleVisitor = mlir::clift::ModuleVisitor<VisitorT>;

template<bool HelperMode>
class mlir::clift::TypeDependencyGraph::Builder
  : public ModuleVisitor<TypeDependencyGraph::Builder<HelperMode>> {
  /// A pointer to the dependency graph being constructed and initialized.
  TypeDependencyGraph *Graph = nullptr;
  uint64_t TargetPointerSize = 0;

  using Base = ModuleVisitor<TypeDependencyGraph::Builder<HelperMode>>;

public:
  Builder(TypeDependencyGraph &Graph, uint64_t TargetPointerSize) :
    Graph(&Graph), TargetPointerSize(TargetPointerSize) {}

  /// Create and initialize a dependency graph.
  static TypeDependencyGraph make(mlir::ModuleOp Module,
                                  uint64_t TargetPointerSize) {
    // Import all the nodes
    TypeDependencyGraph Result;
    Module->walk([&](mlir::Operation *Op) {
      if (auto Global = mlir::dyn_cast<clift::GlobalOpInterface>(Op)) {
        namespace ranks = revng::ranks;
        bool ShouldVisit = pipeline::locationFromString(ranks::HelperFunction,
                                                        Global.getHandle())
                             .has_value();
        if constexpr (not HelperMode)
          ShouldVisit = not ShouldVisit;

        if (ShouldVisit) {
          mlir::LogicalResult MaybeError = Base::visit(Global,
                                                       Result,
                                                       TargetPointerSize);
          revng_assert(MaybeError.succeeded());
        }
      }
    });

    // And all the edges
    for (auto [Type, AssociatedNodes] : Result.TypeToNodes)
      Builder(Result, TargetPointerSize).addDependencies(Type, AssociatedNodes);

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
  revng_log(Log,
            "Added edge: " << getNodeLabel(From) << " --> "
                           << getNodeLabel(To));
  From->addSuccessor(To);
}

using TDG = mlir::clift::TypeDependencyGraph;

template<bool Mode>
void TDG::Builder<Mode>::addNodes(mlir::clift::DefinedType Type) const {
  if (Graph->TypeToNodes.contains(Type)) {
    // There is already a node for this type.
    return;
  }

  constexpr auto Declaration = TypeDefinitionNode::Kind::Declaration;
  auto *DeclNode = Graph->addNode(TypeDefinitionNode{ Type, Declaration });

  revng_log(Log, "Added node: " << getNodeLabel(DeclNode));

  TypeDependencyNode *DefNode = nullptr;
  if (mlir::clift::CEmitter::hasSeparateForwardDeclaration(Type)) {
    constexpr auto Definition = TypeDefinitionNode::Kind::Definition;
    DefNode = Graph->addNode(TypeDefinitionNode{ Type, Definition });

    revng_log(Log, "Added node: " << getNodeLabel(DefNode));

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
  std::optional<mlir::clift::DefinedType> Definition = std::nullopt;
  bool FoundPointer = false;
  bool LastArray = false;
};

static TypeSpecifierResult unwrapType(mlir::clift::ValueType Type,
                                      uint64_t TargetPointerSize) {
  llvm::SmallVector<mlir::clift::TypeStackItem>
    Stack = makeTypeStack(Type, TargetPointerSize);
  revng_assert(!Stack.empty());

  TypeSpecifierResult Result;
  if (auto D = mlir::dyn_cast<mlir::clift::DefinedType>(Stack.back().Type))
    Result.Definition = mlir::cast<mlir::clift::DefinedType>(D);

  if (Stack.size() > 1) {
    using IK = mlir::clift::TypeStackItem::ItemKind;
    Result.LastArray = std::prev(Stack.end(), 2)->Kind == IK::Array;
  }

  Result.FoundPointer = std::ranges::any_of(Stack, [](const auto &Item) {
    using IK = mlir::clift::TypeStackItem::ItemKind;
    return Item.Kind == IK::Pointer;
  });

  return Result;
}

template<bool Mode>
void TDG::Builder<Mode>::addDependenciesFrom(const AssociatedNodes Dependent,
                                             mlir::clift::ValueType DOn) const {
  const auto &[DefinitionDependedOn,
               FoundPointer,
               LastArray] = unwrapType(DOn, TargetPointerSize);

  // If the definition this depends on is not a type definition, we're done,
  // because it cannot be a _real_ dependency. It's either a primitive, an array
  // of primitives, a pointer to a primitive or a primitive-like foreign
  // pointer.
  if (not DefinitionDependedOn)
    return;

  const auto &[DeclarationNode, DefinitionNode] = Dependent;
  revng_assert(DeclarationNode);

  mlir::clift::DefinedType DependentType = DeclarationNode->T;

  TypeDependencyNode *DependentNode = nullptr;
  using CE = mlir::clift::CEmitter;
  bool HasForwardDeclaration = CE::hasSeparateForwardDeclaration(DependentType);
  if (HasForwardDeclaration) {
    revng_assert(DefinitionNode);
    DependentNode = DefinitionNode;
  } else {
    revng_assert(not DefinitionNode);
    DependentNode = DeclarationNode;
  }
  revng_assert(DependentNode);

  auto NodesDependedOn = Graph->TypeToNodes.at(*DefinitionDependedOn);
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
  if (HasForwardDeclaration
      and CE::hasSeparateForwardDeclaration(*DefinitionDependedOn)) {
    addAndLogSuccessor(DependentNode, NodesDependedOn.Definition);
  }

  // Finally, if the `DependentDefinition` has a forward declaration, it also
  // means that it has a separate definition from the forward declaration.
  // In that case, if `DefinitionDependedOn` is a typedef, we also have to look
  // across all those typedefs and ensure the full definition of the dependent
  // also depends on the full definition of the depended-on, across typedefs.
  if (HasForwardDeclaration
      and mlir::isa<mlir::clift::TypedefType>(*DefinitionDependedOn)) {
    auto [Underlying, IsConst] = decomposeTypedef(*DefinitionDependedOn);
    if (auto Defined = mlir::dyn_cast<mlir::clift::DefinedType>(Underlying)) {
      if (CE::hasSeparateForwardDeclaration(Defined)) {
        auto TransitivelyDependedOn = Graph->TypeToNodes.at(Defined);
        revng_assert(TransitivelyDependedOn.Definition);
        addAndLogSuccessor(DependentNode, TransitivelyDependedOn.Definition);
      }
    }
  }
}

template<bool Mode>
void TDG::Builder<Mode>::addDependencies(mlir::clift::DefinedType Type,
                                         AssociatedNodes Nodes) const {
  if (auto Enum = mlir::dyn_cast<mlir::clift::EnumType>(Type)) {
    addDependenciesFrom(Nodes, Enum.getUnderlyingType());

  } else if (auto Struct = mlir::dyn_cast<mlir::clift::StructType>(Type)) {
    for (auto Field : Struct.getFields())
      addDependenciesFrom(Nodes, Field.getType());

  } else if (auto Union = mlir::dyn_cast<mlir::clift::UnionType>(Type)) {
    for (auto Field : Union.getFields())
      addDependenciesFrom(Nodes, Field.getType());

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

template<bool HelperMode>
mlir::LogicalResult TDG::Builder<HelperMode>::visitType(mlir::Type T) {
  if (auto Type = mlir::dyn_cast<mlir::clift::DefinedType>(T)) {
    namespace rr = revng::ranks;
    bool ShouldSkip = pipeline::genericLocationFromString(Type.getHandle(),
                                                          rr::Binary,
                                                          rr::HelperFunction,
                                                          rr::HelperStructType)
                        .has_value();
    if constexpr (HelperMode)
      ShouldSkip = !ShouldSkip;

    if (not ShouldSkip)
      if (auto TD = mlir::dyn_cast<mlir::clift::DefinedType>(Type))
        addNodes(TD);

    if (ShouldSkip)
      dbg << "Skipping: " << Type.getHandle().str() << '\n';
  }

  return mlir::success();
}

mlir::clift::TypeDependencyGraph
mlir::clift::TypeDependencyGraph::makeModelGraph(const mlir::ModuleOp &Module,
                                                 uint64_t TargetPointerSize) {
  return TypeDependencyGraph::Builder<false>::make(Module, TargetPointerSize);
}

mlir::clift::TypeDependencyGraph
mlir::clift::TypeDependencyGraph::makeHelperGraph(const mlir::ModuleOp &Module,
                                                  uint64_t TargetPointerSize) {
  return TypeDependencyGraph::Builder<true>::make(Module, TargetPointerSize);
}

void mlir::clift::TypeDependencyGraph::viewGraph() const {
  llvm::ViewGraph(this, "type-dependency-graph.dot");
}
