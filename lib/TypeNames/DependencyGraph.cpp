//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
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
  }
  return "Invalid";
}

void DependencyGraph::addNode(const model::TypeDefinition *T) {

  constexpr auto Declaration = TypeNode::Kind::Declaration;
  auto *DeclNode = GenericGraph::addNode(TypeNode{ T, Declaration });
  TypeToNode[TypeKindPair{ T, Declaration }] = DeclNode;

  constexpr auto Definition = TypeNode::Kind::Definition;
  auto *DefNode = GenericGraph::addNode(TypeNode{ T, Definition });
  TypeToNode[TypeKindPair{ T, Definition }] = DefNode;
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

struct DependencyEdgeAnalysisResult {
  const model::TypeDefinition *EdgeTarget;
  bool ThereIsAPointerBetweenTypes;
};

static RecursiveCoroutine<DependencyEdgeAnalysisResult>
analyzeDependencyEdges(const model::Type &Type, bool PointerFound = false) {
  if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    const model::Type &Pointee = *Pointer->PointeeType();
    const auto *Defined = llvm::dyn_cast<model::DefinedType>(&Pointee);
    const auto *Definition = Defined ? &Defined->unwrap() : nullptr;
    if (Definition || llvm::isa<model::PrimitiveType>(&Pointee)) {
      rc_return{ .EdgeTarget = Definition,
                 .ThereIsAPointerBetweenTypes = true };
    } else {
      rc_return rc_recur analyzeDependencyEdges(Pointee, true);
    }

  } else if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    const model::Type &Element = *Array->ElementType();
    const auto *Defined = llvm::dyn_cast<model::DefinedType>(&Element);
    const auto *Definition = Defined ? &Defined->unwrap() : nullptr;
    if (Definition || llvm::isa<model::PrimitiveType>(&Element)) {
      rc_return{ .EdgeTarget = Definition,
                 .ThereIsAPointerBetweenTypes = PointerFound };
    } else {
      rc_return rc_recur analyzeDependencyEdges(Element, PointerFound);
    }

  } else {
    // This is only reachable on the very first step.
    const auto *Defined = llvm::dyn_cast<model::DefinedType>(&Type);
    const auto *Definition = Defined ? &Defined->unwrap() : nullptr;
    revng_assert(Definition || llvm::isa<model::PrimitiveType>(&Type));
    rc_return{ .EdgeTarget = Definition, .ThereIsAPointerBetweenTypes = false };
  }
}

template<TypeNode::Kind K>
static TypeDependencyNode *
getDependencyFor(const model::Type &Type,
                 const TypeToDependencyNodeMap &TypeToNode) {

  // TODO: Unfortunately, here we have to deal with some quirks of the C
  // language concerning pointers to arrays of struct/union.
  // Basically, in C, `struct X (*ptr_to_array)[2];` declares a variable
  // `ptr_to_array` that points to an array with two elements of type `struct
  // X`. The problem is that, because of a quirk of paragraph 6.7.6.2 of the
  // C11 standard (Array declarators), to declare `ptr_to_array` it is required
  // to see the complete definition of `struct X`.
  // Even if MSVC seems to compile it just fine, clang and gcc don't.
  //
  // In principle this could be worked around by
  // 1) introducing wrapper structs around arrays of struct/union that are used
  // as pointees
  // 2) postpone the complete definition of the wrapper to after the element
  // type of the array is complete.
  //
  // However for now we just inject a stronger dependency to enforce ordering.
  // This is actually stricter than necessary and can yield to be unable to
  // print valid C code for model that was otherwise perfectly valid and could
  // have been fixed if injected the wrapper structs properly.
  //
  // This particular handling of pointers to array is more strict than actually
  // necessary. It has been implemented as a workaround, instead of handling
  // the emission of wrapper structs. This latter solution of emitting structs
  // has already been used in other places but, in all the other places where we
  // currently do it, it is possible to do it on-the-fly, locally.
  // On the other hand, for dealing with this case properly we'd have to keep
  // track of dependencies between the forward declaration of the wrapper, and
  // the full definition of the element type of the wrapped array.
  // The emission of the full definition of the wrapper must be postponed until
  // the element type of the wrapped type is fully defined, otherwise it would
  // fail compilation. So for now we've put this forced dependency, that could
  // be relaxed if we properly handle the array wrappers.

  DependencyEdgeAnalysisResult Analyzed = analyzeDependencyEdges(Type);
  auto [EdgeTarget, PointerIsBetweenTypes] = Analyzed;
  if (EdgeTarget == nullptr) {
    // By definition, all the primitives are always present.
    // As such, there's no need to add any edges for such cases.
    return nullptr;
  }

  if (llvm::isa<model::ArrayType>(Type)) {
    // If the last type edge was an array, because of the quirks of
    // the C standard mentioned above, we have to depend on the Definition
    // of the element type of the array.
    return TypeToNode.at({ EdgeTarget, TypeNode::Kind::Definition });
  }

  if (PointerIsBetweenTypes) {
    // Otherwise, if we've found at least a pointer, we only depend on the name
    // of the pointee.
    return TypeToNode.at({ EdgeTarget, TypeNode::Kind::Declaration });
  }

  // In all the other cases we depend on the type with the kind indicated by K.
  return TypeToNode.at({ EdgeTarget, K });
}

static void registerDependencies(const model::TypeDefinition &T,
                                 const TypeToDependencyNodeMap &TypeToNode) {

  using Edge = std::pair<TypeDependencyNode *, TypeDependencyNode *>;
  llvm::SmallVector<Edge, 2> Deps;

  // The full definition always depends on declaration.
  // This is not strictly necessary (e.g. when definition and declaration are
  // the same, or when printing a the body of a struct without having forward
  // declared it) but it doesn't introduce cycles and it enables the algorithm
  // that decides on the ordering on the declarations and definitions to make
  // more assumptions about definitions being emitted before declarations.
  auto *DefNode = TypeToNode.at({ &T, TypeNode::Kind::Definition });
  auto *DeclNode = TypeToNode.at({ &T, TypeNode::Kind::Declaration });
  Deps.push_back({ DefNode, DeclNode });

  if (llvm::isa<model::EnumDefinition>(T)) {
    // Enums can only depend on primitives, and those are always present by
    // definition. As such, there's nothing to do here.

  } else if (llvm::isa<model::StructDefinition>(T)
             || llvm::isa<model::UnionDefinition>(T)) {
    // Struct and Union names can always be conjured out of thin air thanks to
    // typedefs. So we only need to add dependencies between their full
    // definition and the full definition of their fields.
    auto *Full = TypeToNode.at({ &T, TypeNode::Kind::Definition });
    for (const model::Type *Edge : T.edges()) {
      if (auto *D = getDependencyFor<TypeNode::Definition>(*Edge, TypeToNode)) {
        Deps.push_back({ Full, D });
        revng_log(Log, getNodeLabel(Full) << " depends on " << getNodeLabel(D));
      }
    }

  } else if (auto *TD = llvm::dyn_cast<model::TypedefDefinition>(&T)) {
    // Typedefs are nasty.
    const model::Type &Under = *TD->UnderlyingType();

    auto *TDDef = TypeToNode.at({ TD, TypeNode::Kind::Definition });
    if (auto *D = getDependencyFor<TypeNode::Definition>(Under, TypeToNode)) {
      Deps.push_back({ TDDef, D });
      revng_log(Log, getNodeLabel(TDDef) << " depends on " << getNodeLabel(D));
    }

    auto *TDDecl = TypeToNode.at({ TD, TypeNode::Kind::Declaration });
    if (auto *D = getDependencyFor<TypeNode::Declaration>(Under, TypeToNode)) {
      Deps.push_back({ TDDecl, D });
      revng_log(Log, getNodeLabel(TDDecl) << " depends on " << getNodeLabel(D));
    }

  } else if (T.isPrototype()) {
    // For function types we can print a valid typedef definition as long as
    // we have visibility on all the names of all the argument types and all
    // return types.

    for (const model::Type *Edge : T.edges()) {
      // The two dependencies added here below are actually stricter than
      // necessary for e.g. stack arguments.
      // The reason is that, on the model, stack arguments are represented by
      // value, but in some cases they are actually passed by pointer in C.
      // Given that with the edges() accessor here we cannot discriminate, we
      // decided to err on the strict side.
      // This could potentially create graphs with loops of dependencies, or
      // make some instances not solvable, that would have otherwise been valid.
      // This should only happen in nasty cases involving loops of function
      // pointers, but possibly other cases we haven't considered.
      // Overall, these remote cases have never showed up until now.
      // If this ever happen, we'll need to fix this properly, either relaxing
      // this dependencies, or pre-processing the model so that what reaches
      // this point is always guaranteed to be in a form that can be emitted.

      if (auto *D = getDependencyFor<TypeNode::Definition>(*Edge, TypeToNode)) {
        Deps.push_back({ DefNode, D });
        revng_log(Log,
                  getNodeLabel(DefNode) << " depends on " << getNodeLabel(D));
      }

      if (auto *D = getDependencyFor<TypeNode::Declaration>(*Edge,
                                                            TypeToNode)) {
        Deps.push_back({ DeclNode, D });
        revng_log(Log,
                  getNodeLabel(DeclNode) << " depends on " << getNodeLabel(D));
      }
    }

  } else {
    revng_abort();
  }

  for (const auto &[From, To] : Deps) {
    revng_log(Log,
              "Adding edge " << getNodeLabel(From) << " --> "
                             << getNodeLabel(To));
    From->addSuccessor(To);
  }
}

DependencyGraph buildDependencyGraph(const TypeVector &Types) {
  DependencyGraph Dependencies;

  // Create nodes
  for (const model::UpcastableTypeDefinition &MT : Types)
    Dependencies.addNode(MT.get());

  // Compute dependencies and add them to the graph
  for (const model::UpcastableTypeDefinition &MT : Types)
    registerDependencies(*MT, Dependencies.TypeNodes());

  if (Log.isEnabled())
    llvm::ViewGraph(&Dependencies, "type-deps.dot");

  return Dependencies;
}
