//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/Support/ModelHelpers.h"

#include "DependencyGraph.h"

static Logger<> Log{ "model-to-header-dependencies" };

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

void DependencyGraph::addNode(const model::Type *T) {

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

template<TypeNode::Kind K>
static TypeDependencyNode *
getDependencyFor(const model::QualifiedType &QT,
                 const TypeToDependencyNodeMap &TypeToNode) {

  // TODO: Unfortunately, here we have to deal with some quirks of the C
  // language concerning pointers to arrays of struct/union.
  // Basically, in C, `struct X (*ptr_to_array)[2];` declares a variable
  // `ptr_to_array` that points to an array with two elements of type `struct
  // X`. The problem is that, because of a quirk of paragraph 6.7.6.2 of the
  // C11 standard (Array declarators), to declare `ptr_to_array` it is required
  // to see the copmlete definition of `struct X`.
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
  // fail compilation. So fow now we've put this forced dependency, that could
  // be relaxed if we properly handle the array wrappers.

  const auto *Unqualified = QT.UnqualifiedType().get();

  bool LastIsArray = false;
  bool PointerFound = false;
  for (const auto &Qualifier : QT.Qualifiers()) {
    if (model::Qualifier::isArray(Qualifier)) {
      LastIsArray = true;
    } else if (model::Qualifier::isPointer(Qualifier)) {
      PointerFound = true;
      LastIsArray = false;
    }
  }

  // If the last non-const qualifier was an array, because of the quirks of the
  // C standard mentioned above, we have to depend on the Definition of the
  // element type of the array.
  if (LastIsArray)
    return TypeToNode.at({ Unqualified, TypeNode::Kind::Definition });

  // Otherwise, if we've found at least a pointer, we only depend on the name of
  // the UnqualifiedType.
  if (PointerFound)
    return TypeToNode.at({ Unqualified, TypeNode::Kind::Declaration });

  // In all the other cases we depend on the UnqualifiedType with the kind
  // indicated by K.
  return TypeToNode.at({ Unqualified, K });
}

static void registerDependencies(const model::Type *T,
                                 const TypeToDependencyNodeMap &TypeToNode) {

  using Edge = std::pair<TypeDependencyNode *, TypeDependencyNode *>;
  llvm::SmallVector<Edge, 2> Deps;

  // The full definition always depends on declaration.
  // This is not strictly necessary (e.g. when definition and declaration are
  // the same, or when printing a the body of a struct without having forward
  // declared it) but it doesn't introduce cycles and it enables the algorithm
  // that decides on the ordering on the declarations and definitions to make
  // more assumptions about definitions being emitted before declarations.
  auto *DefNode = TypeToNode.at({ T, TypeNode::Kind::Definition });
  auto *DeclNode = TypeToNode.at({ T, TypeNode::Kind::Declaration });
  Deps.push_back({ DefNode, DeclNode });

  switch (T->Kind()) {

  case model::TypeKind::Invalid: {
    revng_abort("Primitive or Invalid type should never depend on others");
  } break;

  case model::TypeKind::PrimitiveType: {
    // Nothing to do here. Primitive types names and full definitions can
    // always be defined without dependencies, because they are either not
    // necessary (for primitive types that are already present in stdint.h)
    // or they boil down to a simple typedef of a type in stdint.h. In both
    // cases, the definition provide visibility on both the name and on the
    // full definition.
  } break;

  case model::TypeKind::EnumType: {
    // Enum names and full definitions could always be conjured out of thin
    // air. However, given that we have enums with underlying primitive
    // types, for consistency we enforce that enums names and full
    // definitions always depend on full definition of the underlying
    // primitive type. This adds a little unnecessary edges, but makes the
    // overall structure of the graph easier to reason about. Moreover, full
    // definitions of primitive types can also always be conjured out of
    // thin air, so we're always sure that this does not generates infinite
    // loops.
    const auto *E = cast<model::EnumType>(T);
    const model::QualifiedType &UnderlyingQT = E->UnderlyingType();
    revng_assert(T->edges().size() == 1 and UnderlyingQT == *T->edges().begin()
                 and UnderlyingQT.Qualifiers().empty());

    auto *U = cast<model::PrimitiveType>(UnderlyingQT.UnqualifiedType().get());
    auto *EnumDef = TypeToNode.at({ E, TypeNode::Kind::Declaration });
    auto *EnumDecl = TypeToNode.at({ E, TypeNode::Kind::Definition });
    auto *UnderDecl = TypeToNode.at({ U, TypeNode::Kind::Definition });
    Deps.push_back({ EnumDef, UnderDecl });
    Deps.push_back({ EnumDecl, UnderDecl });
    revng_log(Log,
              getNodeLabel(EnumDef)
                << " depends on " << getNodeLabel(UnderDecl));
    revng_log(Log,
              getNodeLabel(EnumDecl)
                << " depends on " << getNodeLabel(UnderDecl));
  } break;

  case model::TypeKind::StructType:
  case model::TypeKind::UnionType: {
    // Struct and Union names can always be conjured out of thin air thanks to
    // typedefs. So we only need to add dependencies between their full
    // definition and the full definition of their fields.
    auto *Decl = TypeToNode.at({ T, TypeNode::Kind::Definition });
    for (const model::QualifiedType &QT : T->edges()) {
      TypeDependencyNode
        *Dep = getDependencyFor<TypeNode::Definition>(QT, TypeToNode);
      Deps.push_back({ Decl, Dep });
      revng_log(Log, getNodeLabel(Decl) << " depends on " << getNodeLabel(Dep));
    }
  } break;

  case model::TypeKind::TypedefType: {
    // Typedefs are nasty.
    auto *TD = cast<model::TypedefType>(T);
    const model::QualifiedType &Underlying = TD->UnderlyingType();

    auto *TDDef = TypeToNode.at({ TD, TypeNode::Kind::Declaration });
    TypeDependencyNode
      *Def = getDependencyFor<TypeNode::Declaration>(Underlying, TypeToNode);
    Deps.push_back({ TDDef, Def });
    revng_log(Log, getNodeLabel(TDDef) << " depends on " << getNodeLabel(Def));

    auto *TDDecl = TypeToNode.at({ TD, TypeNode::Kind::Definition });
    TypeDependencyNode
      *Decl = getDependencyFor<TypeNode::Definition>(Underlying, TypeToNode);
    Deps.push_back({ TDDecl, Decl });
    revng_log(Log,
              getNodeLabel(TDDecl) << " depends on " << getNodeLabel(Decl));
  } break;

  case model::TypeKind::CABIFunctionType:
  case model::TypeKind::RawFunctionType: {
    // For function types we can print a valid typedef definition as long as
    // we have visibility on all the names of all the argument types and all
    // return types.
    for (const model::QualifiedType &QT : T->edges()) {

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

      TypeDependencyNode
        *Decl = getDependencyFor<TypeNode::Definition>(QT, TypeToNode);
      Deps.push_back({ DefNode, Decl });
      revng_log(Log,
                getNodeLabel(DefNode) << " depends on " << getNodeLabel(Decl));

      TypeDependencyNode
        *Def = getDependencyFor<TypeNode::Declaration>(QT, TypeToNode);
      Deps.push_back({ DeclNode, Def });
      revng_log(Log,
                getNodeLabel(DeclNode) << " depends on " << getNodeLabel(Def));
    }

  } break;

  default:
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
  for (const UpcastablePointer<model::Type> &MT : Types)
    Dependencies.addNode(MT.get());

  // Compute dependencies and add them to the graph
  for (const UpcastablePointer<model::Type> &MT : Types)
    registerDependencies(MT.get(), Dependencies.TypeNodes());

  if (Log.isEnabled())
    llvm::ViewGraph(&Dependencies, "type-deps.dot");

  return Dependencies;
}
