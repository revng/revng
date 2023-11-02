//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

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

#include "DependencyGraph.h"

static Logger<> Log{ "model-to-header-dependencies" };

using namespace llvm;

static llvm::StringRef toString(TypeNode::Kind K) {
  switch (K) {
  case TypeNode::Kind::TypeName:
    return "TypeName";
  case TypeNode::Kind::FullType:
    return "FullType";
  }
  return "Invalid";
}

void DependencyGraph::addNode(const model::Type *T) {

  constexpr auto TypeName = TypeNode::Kind::TypeName;
  auto *NameNode = GenericGraph::addNode(TypeNode{ T, TypeName });
  TypeToNode[TypeKindPair{ T, TypeName }] = NameNode;

  constexpr auto FullType = TypeNode::Kind::FullType;
  auto *FullNode = GenericGraph::addNode(TypeNode{ T, FullType });
  TypeToNode[TypeKindPair{ T, FullType }] = FullNode;
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

static TypeDependencyNode *
getDependencyForTypeName(const model::QualifiedType &QT,
                         const TypeToDependencyNodeMap &TypeToNode) {
  const auto *Unqualified = QT.UnqualifiedType().get();

  // If we find at least a pointer qualifier, then we only need the name of
  // the unqualified type, not its full definition.
  bool ArrayFound = false;
  for (const auto &Qualifier : QT.Qualifiers()) {
    if (model::Qualifier::isPointer(Qualifier))
      return TypeToNode.at({ Unqualified, TypeNode::Kind::TypeName });
    if (model::Qualifier::isArray(Qualifier))
      ArrayFound = true;
  }

  // If we reach this point we haven't found not even a single pointer
  // qualifier.

  // If we did find an array qualifier, we need the full type of the
  // unqualified type.
  if (ArrayFound)
    return TypeToNode.at({ Unqualified, TypeNode::Kind::FullType });

  // Otherwise we can get away with just the name of the unqualified type.
  return TypeToNode.at({ Unqualified, TypeNode::Kind::TypeName });
}

static TypeDependencyNode *
getDependencyForFullType(const model::QualifiedType &QT,
                         const TypeToDependencyNodeMap &TypeToNode) {
  const auto *Unqualified = QT.UnqualifiedType().get();

  // If we find at least a pointer qualifier, then we only need the name of
  // the unqualified type, not its full definition.
  for (const auto &Qualifier : QT.Qualifiers())
    if (model::Qualifier::isPointer(Qualifier))
      return TypeToNode.at({ Unqualified, TypeNode::TypeName });

  // If we reach this point we haven't found not even a single pointer
  // qualifier. Given that we need the full definition, we need the full
  // type of of the unqualified type.
  return TypeToNode.at({ Unqualified, TypeNode::FullType });
}

static void registerDependencies(const model::Type *T,
                                 const TypeToDependencyNodeMap &TypeToNode) {

  using Edge = std::pair<TypeDependencyNode *, TypeDependencyNode *>;
  llvm::SmallVector<Edge, 2> Deps;

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
    auto *EnumName = TypeToNode.at({ E, TypeNode::Kind::TypeName });
    auto *EnumFull = TypeToNode.at({ E, TypeNode::Kind::FullType });
    auto *UnderFull = TypeToNode.at({ U, TypeNode::Kind::FullType });
    Deps.push_back({ EnumName, UnderFull });
    Deps.push_back({ EnumFull, UnderFull });
    revng_log(Log,
              getNodeLabel(EnumName)
                << " depends on " << getNodeLabel(UnderFull));
    revng_log(Log,
              getNodeLabel(EnumFull)
                << " depends on " << getNodeLabel(UnderFull));
  } break;

  case model::TypeKind::StructType:
  case model::TypeKind::UnionType: {
    // Struct and Union names can always be conjured out of thin air thanks to
    // typedefs. So we only need to add dependencies between their full
    // definition and the full definition of their fields.
    auto *Full = TypeToNode.at({ T, TypeNode::Kind::FullType });
    for (const model::QualifiedType &QT : T->edges()) {
      TypeDependencyNode *Dep = getDependencyForFullType(QT, TypeToNode);
      Deps.push_back({ Full, Dep });
      revng_log(Log, getNodeLabel(Full) << " depends on " << getNodeLabel(Dep));
    }
  } break;

  case model::TypeKind::TypedefType: {
    // Typedefs are nasty.
    auto *TD = cast<model::TypedefType>(T);
    const model::QualifiedType &Underlying = TD->UnderlyingType();

    auto *TDName = TypeToNode.at({ TD, TypeNode::Kind::TypeName });
    TypeDependencyNode *NameDep = getDependencyForTypeName(Underlying,
                                                           TypeToNode);
    Deps.push_back({ TDName, NameDep });
    revng_log(Log,
              getNodeLabel(TDName) << " depends on " << getNodeLabel(NameDep));

    auto *TDFull = TypeToNode.at({ TD, TypeNode::Kind::FullType });
    TypeDependencyNode *FullDep = getDependencyForFullType(Underlying,
                                                           TypeToNode);
    Deps.push_back({ TDFull, FullDep });
    revng_log(Log,
              getNodeLabel(TDFull) << " depends on " << getNodeLabel(FullDep));
  } break;

  case model::TypeKind::CABIFunctionType:
  case model::TypeKind::RawFunctionType: {
    // For function types we can print a valid typedef definition as long as
    // we have visibility on all the names of all the argument types and all
    // return types.
    auto *FullNode = TypeToNode.at({ T, TypeNode::Kind::FullType });
    auto *NameNode = TypeToNode.at({ T, TypeNode::Kind::TypeName });
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

      TypeDependencyNode *FullDep = getDependencyForFullType(QT, TypeToNode);
      Deps.push_back({ FullNode, FullDep });

      TypeDependencyNode *NameDep = getDependencyForTypeName(QT, TypeToNode);
      Deps.push_back({ NameNode, NameDep });

      revng_log(Log,
                getNodeLabel(FullNode)
                  << " depends on " << getNodeLabel(FullDep));
      revng_log(Log,
                getNodeLabel(NameNode)
                  << " depends on " << getNodeLabel(NameDep));
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
