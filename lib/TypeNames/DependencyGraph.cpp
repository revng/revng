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

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Generated/ForwardDecls.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/TypeNames/DependencyGraph.h"

static Logger<> Log{ "type-dependency-graph" };

using namespace llvm;

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
  const model::UpcastableType &NodeType = N->T;
  revng_assert(not NodeType.empty());
  if (const model::TypeDefinition *TD = NodeType->tryGetAsDefinition()) {
    return (Twine(getNameFromYAMLScalar(TD->key())) + Twine("-")
            + Twine(toString(N->K)))
      .str();
  }
  return (Twine(NodeType->toString()) + Twine("-") + Twine(toString(N->K)))
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

  /// A pointer to the model::Binary for which the Builder is building a
  /// DependencyGraph.
  const model::Binary *TheBinary = nullptr;

public:
  Builder(const model::Binary &B) : Graph(nullptr), TheBinary(&B) {}

  // Ensure we don't initialize TheBinary to the address of a temporary.
  Builder(model::Binary &&) = delete;

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

  /// Create and initialize a DependencyGraph from a model::Binary.
  static DependencyGraph make(const model::Binary &B) {
    return Builder(B).make();
  }

private:
  /// Actual implementation of the make method.
  void makeImpl() const;

  /// Adds a declaration node and a definition node to Graph for \p T and
  /// returns their AssociatedNodes.
  AssociatedNodes addNodes(const model::TypeDefinition &T) const;

  /// Adds a declaration node and a definition node to Graph for an artificial
  /// struct wrapper intended to wrap \p T, and returns their AssociatedNodes.
  /// \p T is required to have nonzero size.
  AssociatedNodes addArtificialNodes(const model::Type &T) const;

  /// Adds a declaration node and a definition node to Graph for an artificial
  /// struct wrapper intended to wrap \p T, and return their AssociatedNodes.
  /// \p T is required to return a RegisterSet, and the artificial wrapper is a
  /// struct with each of those registers' types as fields.
  AssociatedNodes
  addArtificialNodes(const model::RawFunctionDefinition &T) const;

  /// Adds all the necessary dependency edges to Graph from the \p Dependent
  /// AssociatedNodes to the \p DependedOn model::Type.
  /// This may include adding new struct wrapper nodes, along with dependency
  /// edges from and to them.
  void addDependenciesFrom(const AssociatedNodes Dependent,
                           const model::Type &DependedOn) const;

  /// Adds all the necessary dependency edges to Graph for the nodes that
  /// represent the declaration and definition of \p T.
  /// This may include adding new struct wrapper nodes, along with dependency
  /// edges from and to them.
  void addDependencies(const model::TypeDefinition &T) const;

  /// Given a model::ArrayType \p Array that needs to be wrapped into an
  /// artificial struct wrapper, adds a declaration and a definition node for
  /// the wrapper, along with all the necessary dependency edges from and to the
  /// wrapped types.
  /// Returns the AssociatedNodes of the wrapper.
  AssociatedNodes
  addWrappedArrayWithDependencies(const model::ArrayType &Array) const;

  /// Given a model::RawFunctionDefinition \p RF returning a RegisterSet, adds a
  /// declaration and a definition node for an artificial struct wrapper
  /// whose fields are all the return register types, along with all the
  /// necessary dependency edges from and to the wrapped types.
  /// Returns the AssociatedNodes of the wrapper.
  AssociatedNodes
  addWrappedRFWithDependencies(const model::RawFunctionDefinition &RF) const;
};

using DGBuilder = DependencyGraph::Builder;

static_assert(std::is_copy_constructible_v<DGBuilder>);
static_assert(std::is_copy_assignable_v<DGBuilder>);
static_assert(std::is_move_constructible_v<DGBuilder>);
static_assert(std::is_move_assignable_v<DGBuilder>);

static void addAndLogSuccessor(TypeDependencyNode *From,
                               TypeDependencyNode *To) {
  revng_log(Log,
            "Adding edge " << getNodeLabel(From) << " --> "
                           << getNodeLabel(To));
  From->addSuccessor(To);
}

DependencyGraph::AssociatedNodes
DGBuilder::addNodes(const model::TypeDefinition &T) const {

  using TypeNodeGenericGraph = GenericGraph<TypeDependencyNode>;
  auto *G = static_cast<TypeNodeGenericGraph *const>(Graph);
  const model::UpcastableType UT = TheBinary->makeType(T.key());

  constexpr auto Declaration = TypeNode::Kind::Declaration;
  auto *DeclNode = G->addNode(TypeNode{ UT, Declaration });

  constexpr auto Definition = TypeNode::Kind::Definition;
  auto *DefNode = G->addNode(TypeNode{ UT, Definition });

  // The definition always depends on the declaration.
  // This is not strictly necessary (e.g. when definition and declaration are
  // the same, or when printing a the body of a struct without having forward
  // declared it) but it doesn't introduce cycles and it enables the algorithm
  // that decides on the ordering on the declarations and definitions to make
  // more assumptions about definitions being emitted before declarations.
  addAndLogSuccessor(DefNode, DeclNode);

  return Graph->TypeToNodes[&T] = AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

DependencyGraph::AssociatedNodes
DGBuilder::addArtificialNodes(const model::Type &T) const {

  const auto UT = model::UpcastableType(T);
  if (auto It = Graph->WrappedToNodes.find(UT);
      It != Graph->WrappedToNodes.end()) {
    return It->second;
  }

  using TypeNodeGenericGraph = GenericGraph<TypeDependencyNode>;
  auto *G = static_cast<TypeNodeGenericGraph *const>(Graph);

  constexpr auto Declaration = TypeNode::Kind::ArtificialWrapperDeclaration;
  auto *DeclNode = G->addNode(TypeNode{ UT, Declaration });

  constexpr auto Definition = TypeNode::Kind::ArtificialWrapperDefinition;
  auto *DefNode = G->addNode(TypeNode{ UT, Definition });

  // The definition always depends on the declaration.
  // This is not strictly necessary but it doesn't introduce cycles and it
  // enables the algorithm that decides on the ordering on the declarations and
  // definitions to make more assumptions about definitions being emitted before
  // declarations.
  addAndLogSuccessor(DefNode, DeclNode);

  return Graph->WrappedToNodes[UT] = AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

DependencyGraph::AssociatedNodes
DGBuilder::addArtificialNodes(const model::RawFunctionDefinition &T) const {

  const model::UpcastableType UT = TheBinary->makeType(T.key());
  if (auto It = Graph->WrappedToNodes.find(UT);
      It != Graph->WrappedToNodes.end()) {
    return It->second;
  }

  auto Layout = abi::FunctionType::Layout::make(T);
  using namespace abi::FunctionType::ReturnMethod;
  revng_assert(Layout.returnMethod() == RegisterSet);

  using TypeNodeGenericGraph = GenericGraph<TypeDependencyNode>;
  auto *G = static_cast<TypeNodeGenericGraph *const>(Graph);

  constexpr auto Declaration = TypeNode::Kind::ArtificialWrapperDeclaration;
  auto *DeclNode = G->addNode(TypeNode{ UT, Declaration });

  constexpr auto Definition = TypeNode::Kind::ArtificialWrapperDefinition;
  auto *DefNode = G->addNode(TypeNode{ UT, Definition });

  // The definition always depends on the declaration.
  // This is not strictly necessary (e.g. when definition and declaration are
  // the same, or when printing a the body of a struct without having forward
  // declared it) but it doesn't introduce cycles and it enables the algorithm
  // that decides on the ordering on the declarations and definitions to make
  // more assumptions about definitions being emitted before declarations.
  addAndLogSuccessor(DefNode, DeclNode);

  return Graph->WrappedToNodes[UT] = AssociatedNodes{
    .Declaration = DeclNode,
    .Definition = DefNode,
  };
}

struct TypeSpecifierResult {
  const model::TypeDefinition *SpecifierDefinition;
  bool FoundPointer;
};

/// Returns the type specifier of a model::Type \p T if it is a
/// model::TypeDefinition. If it's a primitive type returns nullptr.
/// The FoundPointer field of the return value is set to true is at any point in
/// the traversal from T to the SpecifierDefinition a model::PointerType was
/// encountered.
static RecursiveCoroutine<TypeSpecifierResult>
getTypeSpecifierResult(const model::Type &T, bool FoundPointer = false) {

  if (const auto *P = dyn_cast<model::PointerType>(&T)) {
    const model::Type &Pointee = *P->PointeeType();
    rc_return rc_recur getTypeSpecifierResult(Pointee, true);
  }

  if (auto *A = dyn_cast<model::ArrayType>(&T)) {
    const model::Type &Element = *A->ElementType();
    rc_return rc_recur getTypeSpecifierResult(Element, FoundPointer);
  }

  if (auto *Definition = T.tryGetAsDefinition()) {
    rc_return{ Definition, FoundPointer };
  }

  rc_return{ nullptr, FoundPointer };
}

/// Returns the type specifier of a model::Type \p T if it is a
/// model::TypeDefinition. If it's a primitive type returns nullptr.
static const model::TypeDefinition *
getTypeSpecifierDefinition(const model::Type &T) {
  TypeSpecifierResult Result = getTypeSpecifierResult(T, false);
  return Result.SpecifierDefinition;
}

static auto returnValueTypes(const abi::FunctionType::Layout &L) {
  using abi::FunctionType::Layout;
  return llvm::map_range(L.ReturnValues,
                         [](const Layout::ReturnValue &RV) { return RV.Type; });
}

static RecursiveCoroutine<const model::Type *>
skipPointers(const model::PointerType &Pointer) {
  const model::Type &Pointee = *Pointer.PointeeType();
  if (auto *Ptr = dyn_cast<model::PointerType>(&Pointee))
    rc_return rc_recur skipPointers(*Ptr);

  rc_return &Pointee;
}

/// If \p T is not a model::PointerType returns nullptr.
/// If \p is a model::PointerType and its pointee type is also a PointerType,
/// traverses the pointees until it finds a non-pointer.
/// If that non-pointer is a model::ArrayType, returns it, otherwise returns a
/// nullptr.
static RecursiveCoroutine<const model::ArrayType *>
getArrayPointee(const model::Type &T) {

  const auto *Pointer = dyn_cast<model::PointerType>(&T);
  if (not Pointer)
    rc_return nullptr;

  const model::Type *Pointee = skipPointers(*Pointer);

  if (const auto *Array = dyn_cast<model::ArrayType>(Pointee))
    rc_return Array;

  rc_return nullptr;
}

DependencyGraph::AssociatedNodes
DGBuilder::addWrappedArrayWithDependencies(const model::ArrayType &Array)
  const {

  AssociatedNodes ArtificialWrapper = addArtificialNodes(Array);
  const auto &[WrapperDecl, WrapperDef] = ArtificialWrapper;

  // WrapperDecl has no dependencies, because it's the declaration node for a
  // struct wrapper, and struct can always be forward declared without
  // depending on any types. WrapperDef, instead, needs to depend on the full
  // definition of the type specifier of the array that the dependent depends
  // on, i.e. the SpecifierDef.
  const model::TypeDefinition
    *TypeSpecifier = getTypeSpecifierDefinition(Array);
  const auto &[SpecifierDecl,
               SpecifierDef] = Graph->TypeToNodes.at(TypeSpecifier);
  addAndLogSuccessor(WrapperDef, SpecifierDef);

  return ArtificialWrapper;
}

void DGBuilder::addDependenciesFrom(const AssociatedNodes Dependent,
                                    const model::Type &DependedOn) const {

  const auto &[DependentDecl, DependentDef] = Dependent;

  const auto AddArrayWrapper = [&](const model::ArrayType &Array) {
    const auto &[ArrayWrapperDecl,
                 ArrayWrapperDef] = addWrappedArrayWithDependencies(Array);
    // The declaration of Dependent always depends on the declaration of the
    // struct wrapper that wraps the Array.
    addAndLogSuccessor(DependentDecl, ArrayWrapperDecl);
    // The definition of Dependent also only depends on the declaration of the
    // struct wrapper that wraps the Array.
    // The reason is that Dependent will only contain a pointer to the struct
    // wrapper that wraps the Array, so it only needs the forward declaration
    // of the wrapper.
    addAndLogSuccessor(DependentDef, ArrayWrapperDecl);
  };

  // If the DependedOn is a pointer-to-array, wrap the pointed array in a struct
  // wrapper, creating the necessary artificial nodes and dependencies.
  if (const model::ArrayType *Array = getArrayPointee(DependedOn)) {
    AddArrayWrapper(*Array);
    return;
  }

  bool Artificial = DependentDecl->isArtificial();
  revng_assert(Artificial == DependentDef->isArtificial());

  const model::TypeDefinition *TD = DependentDecl->T->tryGetAsDefinition();
  bool IsFunction = isa<model::RawFunctionDefinition>(TD)
                    or isa<model::CABIFunctionDefinition>(TD);
  const model::ArrayType *Array = dyn_cast<model::ArrayType>(&DependedOn);
  if (Artificial and IsFunction and Array) {
    AddArrayWrapper(*Array);
    return;
  }

  TypeSpecifierResult Result = getTypeSpecifierResult(DependedOn);
  const auto &[TypeSpecifierDef, FoundPointer] = Result;
  if (not TypeSpecifierDef)
    return;

  AssociatedNodes Specifier = Graph->TypeToNodes.at(TypeSpecifierDef);
  // The declaration of Dependent needs the declaration of the Specifier only if
  // the declaration of Dependent is *not* a forward declaration.
  // If the declaration of Dependent is a forward declaration we can avoi adding
  // a dependency from DependentDecl to Specifier.Declaration
  if (not isa<model::StructDefinition>(TD)
      and not isa<model::UnionDefinition>(TD)
      and not isa<model::EnumDefinition>(TD))
    addAndLogSuccessor(DependentDecl, Specifier.Declaration);

  // If we found a pointer on the path to TypeSpecifierDef, the definition of
  // Dependent only needs the declaration of the Specifier.
  // Otherwise it needs the full definition of the Specifier.
  if (FoundPointer)
    addAndLogSuccessor(DependentDef, Specifier.Declaration);
  else
    addAndLogSuccessor(DependentDef, Specifier.Definition);
}

DependencyGraph::AssociatedNodes
DGBuilder::addWrappedRFWithDependencies(const model::RawFunctionDefinition &RF)
  const {
  AssociatedNodes ReturnValuesWrapper = addArtificialNodes(RF);
  const auto &[RVWrapperDecl, RVWrapperDef] = ReturnValuesWrapper;

  using abi::FunctionType::Layout;
  auto Layout = Layout::make(RF);

  // Then the artificial wrapper wrapping the return values should depend on the
  // relevant return values types, that are the types of its fields.
  auto ReturnValues = returnValueTypes(Layout);
  revng_assert(not ReturnValues.empty());
  for (const model::UpcastableType &ReturnType : ReturnValues) {
    revng_assert(not ReturnType->isArray());
    addDependenciesFrom(ReturnValuesWrapper, *ReturnType);
  }

  return ReturnValuesWrapper;
}

void DGBuilder::addDependencies(const model::TypeDefinition &T) const {

  const auto &TDNodes = Graph->TypeToNodes.at(&T);
  const auto &[DeclNode, DefNode] = TDNodes;

  switch (T.Kind()) {

  case model::TypeDefinitionKind::TypedefDefinition:
  case model::TypeDefinitionKind::EnumDefinition:
  case model::TypeDefinitionKind::StructDefinition:
  case model::TypeDefinitionKind::UnionDefinition: {
    for (const model::Type *Edge : T.edges())
      addDependenciesFrom(TDNodes, *Edge);
  } break;

  case model::TypeDefinitionKind::RawFunctionDefinition:
  case model::TypeDefinitionKind::CABIFunctionDefinition: {
    // Function types may require to add struct wrappers around argument types
    // and return types.
    using abi::FunctionType::Layout;
    auto Layout = Layout::make(T);

    using namespace abi::FunctionType::ReturnMethod;
    if (Layout.returnMethod() == RegisterSet) {

      // If T is a RawFunctionDefinition returning a RegisterSet, we create an
      // artificial struct wrapper around the returned registers.
      const auto *RF = cast<model::RawFunctionDefinition>(&T);
      AssociatedNodes WrapperNodes = addWrappedRFWithDependencies(*RF);

      // Both the declaration and the definition of the function type depend
      // only on the declaration of the struct wrapper for the return type.
      addAndLogSuccessor(DeclNode, WrapperNodes.Declaration);
      addAndLogSuccessor(DefNode, WrapperNodes.Declaration);
    } else {
      for (const model::UpcastableType &ReturnType : returnValueTypes(Layout)) {
        addDependenciesFrom(TDNodes, *ReturnType);
      }
    }

    auto ArgumentTypes = llvm::map_range(Layout.Arguments,
                                         [](const Layout::Argument &A) {
                                           return A.Type;
                                         });
    for (const model::UpcastableType &ArgumentType : ArgumentTypes)
      addDependenciesFrom(TDNodes, *ArgumentType);

  } break;

  default:
    revng_abort("Unexpected T.Kind()");
  }
}

void DGBuilder::makeImpl() const {

  // Create declaration and definition nodes for all the nodes
  for (const model::UpcastableTypeDefinition &MT : TheBinary->TypeDefinitions())
    addNodes(*MT);

  // Compute dependencies and add them to the graph
  for (const model::UpcastableTypeDefinition &MT : TheBinary->TypeDefinitions())
    addDependencies(*MT);
}

DependencyGraph DependencyGraph::make(const model::Binary &B) {
  return DGBuilder::make(B);
}
