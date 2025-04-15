//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/TypeNames/DependencyGraph.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

static Logger<> Log{ "type-dependency-graph" };

using namespace llvm;

static bool hasSeparateForwardDeclaration(const model::TypeDefinition &TD) {
  return not ptml::CTypeBuilder::isDeclarationTheSameAsDefinition(TD);
}

static llvm::StringRef toString(TypeNode::Kind K) {
  switch (K) {
  case TypeNode::Kind::Declaration:
    return "Declaration";
  case TypeNode::Kind::Definition:
    return "Definition";
  case TypeNode::Kind::ArtificialWrapperDeclaration:
    return "ArtificialWrapperDeclaration";
  case TypeNode::Kind::ArtificialWrapperDefinition:
    return "ArtificialWrapperDefinition";
  }
  return "Invalid";
}

std::string getNodeLabel(const TypeDependencyNode *N) {
  return (Twine(getNameFromYAMLScalar(N->T->key())) + Twine("-")
          + Twine(toString(N->K)))
    .str();
}

using DepNode = TypeDependencyNode;
using DepGraph = DependencyGraph;
std::string llvm::DOTGraphTraits<DepGraph *>::getNodeLabel(const DepNode *N,
                                                           const DepGraph *G) {
  return ::getNodeLabel(N);
}

class DependencyGraph::Builder {
  /// A pointer to the DependencyGraph being constructed and initialized.
  DependencyGraph *Graph = nullptr;

  /// A pointer to the TypeVector for which the Builder is building a
  /// DependencyGraph.
  const TypeVector *Types = nullptr;

public:
  Builder(const TypeVector &TV) : Graph(nullptr), Types(&TV) {}

  // Ensure we don't initialize Types to the address of a temporary.
  Builder(TypeVector &&TV) = delete;

public:
  /// Create and initialize a DependencyGraph.
  DependencyGraph make() {

    // Set up an empty DependencyGraph, and the Graph pointer to point to it, so
    // that all methods that are used to build the graph from makeImpl down can
    // just use Graph.
    // The Graph pointer is then reset to nullptr via the ScopedExchange when
    // construction is done.
    DependencyGraph Dependencies;
    ScopedExchange ExchangeGraphPtr(Graph, &Dependencies);

    makeImpl();

    if (Log.isEnabled())
      llvm::ViewGraph(Graph, "type-deps.dot");

    return Dependencies;
  }

  /// Create and initialize a DependencyGraph from a TypeVector.
  static DependencyGraph make(const TypeVector &TV) {
    return Builder(TV).make();
  }

private:
  /// Actual implementation of the make method.
  void makeImpl() const;

  /// Add a declaration node and a definition node to Graph for \p T.
  AssociatedNodes addNodes(const model::TypeDefinition &T) const;

  /// Add a declaration node and a definition node to Graph for an artificial
  /// struct wrapper intended to wrap \p T. \p T is required to return a
  /// RegisterSet, and the artificial wrapper is a struct with each of those
  /// registers' types as fields.
  AssociatedNodes
  addArtificialNodes(const model::RawFunctionDefinition &T) const;

  /// Add all the necessary dependency edges to Graph for the nodes that
  /// represent the declaration and definition of \p T.
  void addDependencies(const model::TypeDefinition &T) const;

  void addDependenciesFrom(const AssociatedNodes Dependent,
                           const model::Type &DependedOn) const;

  AssociatedNodes
  addRFTReturnWrapper(const model::RawFunctionDefinition &RFD) const;
};

static void addAndLogSuccessor(TypeDependencyNode *From,
                               TypeDependencyNode *To) {
  revng_assert(From);
  revng_assert(To);
  revng_log(Log,
            "Adding edge " << getNodeLabel(From) << " --> "
                           << getNodeLabel(To));
  From->addSuccessor(To);
}

DependencyGraph::AssociatedNodes
DependencyGraph::Builder::addNodes(const model::TypeDefinition &T) const {

  constexpr auto Declaration = TypeNode::Kind::Declaration;
  auto *DeclNode = Graph->addNode(TypeNode{ &T, Declaration });

  revng_log(Log, "Added DeclNode node " << getNodeLabel(DeclNode));

  TypeDependencyNode *DefNode = nullptr;
  if (hasSeparateForwardDeclaration(T)) {
    constexpr auto Definition = TypeNode::Kind::Definition;
    DefNode = Graph->addNode(TypeNode{ &T, Definition });

    revng_log(Log, "Added DefNode node " << getNodeLabel(DefNode));

    // The definition always depends on the declaration.
    // This is not strictly necessary (e.g. when definition and declaration are
    // the same, or when printing a the body of a struct without having forward
    // declared it) but it doesn't introduce cycles and it enables the algorithm
    // that decides on the ordering on the declarations and definitions to make
    // more assumptions about definitions being emitted before declarations.
    addAndLogSuccessor(DefNode, DeclNode);
  }

  revng_assert(not Graph->TypeToNodes.contains(&T));
  revng_log(Log, "Added DeclNode: " << DeclNode << ", DefNode: " << DefNode);
  return Graph->TypeToNodes[&T] = AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

DependencyGraph::AssociatedNodes
DependencyGraph::Builder::addArtificialNodes(const model::RawFunctionDefinition
                                               &T) const {

  auto Layout = abi::FunctionType::Layout::make(T);
  using namespace abi::FunctionType::ReturnMethod;
  revng_assert(Layout.returnMethod() == RegisterSet);

  constexpr auto Declaration = TypeNode::Kind::ArtificialWrapperDeclaration;
  auto *DeclNode = Graph->addNode(TypeNode{ &T, Declaration });
  revng_log(Log, "Added DeclNode node " << getNodeLabel(DeclNode));

  constexpr auto Definition = TypeNode::Kind::ArtificialWrapperDefinition;
  auto *DefNode = Graph->addNode(TypeNode{ &T, Definition });
  revng_log(Log, "Added DefNode node " << getNodeLabel(DefNode));

  // The definition always depends on the declaration.
  // This is not strictly necessary (e.g. when definition and declaration are
  // the same, or when printing a the body of a struct without having forward
  // declared it) but it doesn't introduce cycles and it enables the algorithm
  // that decides on the ordering on the declarations and definitions to make
  // more assumptions about definitions being emitted before declarations.
  addAndLogSuccessor(DefNode, DeclNode);

  return AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

struct TypeSpecifierResult {
  const model::TypeDefinition *Definition;
  bool FoundPointer;
  bool LastArray;
};

static RecursiveCoroutine<TypeSpecifierResult>
getTypeSpecifierResult(const model::Type &T,
                       bool FoundPointer = false,
                       bool LastArray = false) {
  if (const auto *P = dyn_cast<model::PointerType>(&T)) {
    const model::Type &Pointee = *P->PointeeType();
    rc_return rc_recur getTypeSpecifierResult(Pointee, true, false);
  }

  if (const auto *A = dyn_cast<model::ArrayType>(&T)) {
    const model::Type &Element = *A->ElementType();
    rc_return rc_recur getTypeSpecifierResult(Element, FoundPointer, true);
  }

  if (const auto *D = T.tryGetAsDefinition()) {
    rc_return TypeSpecifierResult{ .Definition = D,
                                   .FoundPointer = FoundPointer,
                                   .LastArray = LastArray };
  }

  rc_return TypeSpecifierResult{ .Definition = nullptr,
                                 .FoundPointer = FoundPointer,
                                 .LastArray = LastArray };
}

void DependencyGraph::Builder::addDependenciesFrom(const AssociatedNodes
                                                     Dependent,
                                                   const model::Type
                                                     &DependedOn) const {
  TypeSpecifierResult SpecifierDependedOn = getTypeSpecifierResult(DependedOn);
  const auto &[DefinitionDependedOn,
               FoundPointer,
               LastArray] = SpecifierDependedOn;

  // If the DefinitionDependedOn is not a TypeDefinition, we're done, because it
  // hasn't any real dependency, except on primitives, arrays of primitives, and
  // pointers to primitives.
  if (not DefinitionDependedOn)
    return;

  const auto &[DependentDeclNode, DependentDefNode] = Dependent;
  const model::TypeDefinition *DependentDefinition = DependentDeclNode->T;

  revng_assert(DependentDeclNode);
  revng_assert(DependentDefinition);

  TypeDependencyNode *DependentNode = nullptr;
  bool
    HasForwardDeclaration = hasSeparateForwardDeclaration(*DependentDefinition)
                            or DependentDeclNode->isArtificial();
  if (HasForwardDeclaration) {
    revng_assert(DependentDefNode);
    DependentNode = DependentDefNode;
  } else {
    revng_assert(not DependentDefNode);
    DependentNode = DependentDeclNode;
  }
  revng_assert(DependentNode);

  AssociatedNodes NodesDependedOn = Graph->TypeToNodes.at(DefinitionDependedOn);
  revng_assert(NodesDependedOn.Declaration);
  revng_assert(not NodesDependedOn.Declaration->isArtificial());
  revng_assert(not NodesDependedOn.Definition
               or not NodesDependedOn.Definition->isArtificial());

  // If LastArray is true, the node depended on is always the node
  // representing the full definition of the type, which in some cases might
  // be the Declaration node (e.g. when the type depended on doesn't have a
  // separate Definition and Declaration node, but only a Declaration, such as
  // for TypedefDefinitions and function type definitions).
  if (LastArray) {
    TypeDependencyNode *NodeDependedOn = NodesDependedOn.Definition ?
                                           NodesDependedOn.Definition :
                                           NodesDependedOn.Declaration;
    revng_assert(NodeDependedOn);
    addAndLogSuccessor(DependentNode, NodeDependedOn);
    return;
  }

  // If FoundPointer is true, and LastArray is false, the node depended on is
  // always the Declaration node, because we don't need the full definition.
  if (FoundPointer) {
    addAndLogSuccessor(DependentNode, NodesDependedOn.Declaration);
    return;
  }

  // Otherwise we fall back in the baseline case.

  // The DependentNode always depends on the the Declaration node of DependedOn.
  addAndLogSuccessor(DependentNode, NodesDependedOn.Declaration);

  // If both Dependent and DependedOn has a separate forward declaration we add
  // a dependency from DependentNode to the Definition of the DependedOn.
  if (HasForwardDeclaration
      and hasSeparateForwardDeclaration(*DefinitionDependedOn)) {
    addAndLogSuccessor(DependentNode, NodesDependedOn.Definition);
  }

  // Finally, if the DependentDefinition has a forward declaration, it also
  // means that it has a separate definition from the forward declaration.
  // In that case, if DefinitionDependedOn is a typedef, we also have to look
  // across all those typedefs and ensure the full definition of the dependent
  // also depends on the full definition of the depended-on, across typedefs.
  if (HasForwardDeclaration
      and isa<model::TypedefDefinition>(DefinitionDependedOn)) {
    const model::Type *Underlying = DefinitionDependedOn->skipTypedefs();
    if (auto *UnderlyingDefinition = Underlying->tryGetAsDefinition();
        UnderlyingDefinition
        and hasSeparateForwardDeclaration(*UnderlyingDefinition)) {

      AssociatedNodes NodesTransitivelyDependedOn = Graph->TypeToNodes
                                                      .at(UnderlyingDefinition);
      revng_assert(NodesTransitivelyDependedOn.Definition);
      addAndLogSuccessor(DependentNode, NodesTransitivelyDependedOn.Definition);
    }
  }
}

DependencyGraph::AssociatedNodes
DependencyGraph::Builder::addRFTReturnWrapper(const model::RawFunctionDefinition
                                                &RFD) const {
  AssociatedNodes ReturnValuesWrapper = addArtificialNodes(RFD);
  const auto &[WrapperDecl, WrapperDef] = ReturnValuesWrapper;

  auto Layout = abi::FunctionType::Layout::make(RFD);

  // Then the artificial wrapper wrapping the return values should depend on the
  // relevant return value types, that are the types of its fields.
  for (const model::UpcastableType &ReturnType : Layout.returnValueTypes()) {
    addDependenciesFrom(ReturnValuesWrapper, *ReturnType);
  }
  return ReturnValuesWrapper;
}

void DependencyGraph::Builder::addDependencies(const model::TypeDefinition &T)
  const {

  const auto &TDNodes = Graph->TypeToNodes.at(&T);

  switch (T.Kind()) {

  case model::TypeDefinitionKind::EnumDefinition:
  case model::TypeDefinitionKind::StructDefinition:
  case model::TypeDefinitionKind::UnionDefinition:
  case model::TypeDefinitionKind::TypedefDefinition: {
    for (const model::Type *Edge : T.edges())
      addDependenciesFrom(TDNodes, *Edge);
  } break;

  case model::TypeDefinitionKind::RawFunctionDefinition:
  case model::TypeDefinitionKind::CABIFunctionDefinition: {
    using abi::FunctionType::Layout;
    auto TheLayout = Layout::make(T);

    using namespace abi::FunctionType::ReturnMethod;
    if (TheLayout.returnMethod() != RegisterSet) {
      for (const model::UpcastableType &ReturnType :
           TheLayout.returnValueTypes())
        addDependenciesFrom(TDNodes, *ReturnType);
    } else {
      // If T is a RawFunctionDefinition returning a RegisterSet, we create an
      // artificial struct wrapper around the returned registers.
      const auto *RF = cast<model::RawFunctionDefinition>(&T);
      AssociatedNodes WrapperNodes = addRFTReturnWrapper(*RF);

      // The declaration of the function type depends only on the declaration of
      // the struct wrapper for the return type.
      addAndLogSuccessor(TDNodes.Declaration, WrapperNodes.Declaration);
      revng_assert(not TDNodes.Definition);
    }

    for (const model::UpcastableType &ArgumentType : TheLayout.argumentTypes())
      addDependenciesFrom(TDNodes, *ArgumentType);

  } break;

  default:
    revng_abort("Undexpected T.Kind()");
  }
}

void DependencyGraph::Builder::makeImpl() const {

  // Create declaration and definition nodes for all the type definitions
  for (const model::UpcastableTypeDefinition &MT : *Types)
    addNodes(*MT);

  // Compute dependencies and add them to the graph
  for (const model::UpcastableTypeDefinition &MT : *Types)
    addDependencies(*MT);
}

DependencyGraph DependencyGraph::make(const TypeVector &TV) {

  static_assert(std::is_copy_constructible_v<DependencyGraph::Builder>);
  static_assert(std::is_copy_assignable_v<DependencyGraph::Builder>);
  static_assert(std::is_move_constructible_v<DependencyGraph::Builder>);
  static_assert(std::is_move_assignable_v<DependencyGraph::Builder>);

  return DependencyGraph::Builder::make(TV);
}
