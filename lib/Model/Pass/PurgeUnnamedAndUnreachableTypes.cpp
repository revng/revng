/// \file PurgeUnnamedAndUnreachableTypes.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/Pass/RegisterModelPass.h"

using namespace llvm;

static RegisterModelPass R("purge-unnamed-and-unreachable-types",
                           "Remove all the types that cannot be reached from "
                           "any named type or a type \"outside\" the type "
                           "system itself",
                           model::purgeUnnamedAndUnreachableTypes);

static RegisterModelPass R2("purge-unreachable-types",
                            "Remove all the types that cannot be reached from "
                            "a type \"outside\" the type system itself",
                            model::purgeUnreachableTypes);

namespace model {
static void purgeTypesImpl(TupleTree<model::Binary> &Model,
                           bool KeepTypesWithName);
}

template<typename V, typename T, size_t... Indices>
static void
visitTuple(V &&Visitor, T &Tuple, const std::index_sequence<Indices...> &) {
  (Visitor(get<Indices>(Tuple)), ...);
}

template<typename V, TupleLike T>
static void visitTuple(V &&Visitor, T &Tuple) {
  visitTuple(std::forward<V>(Visitor),
             Tuple,
             std::make_index_sequence<std::tuple_size_v<T>>{});
}

template<typename V, TupleLike T, typename E>
static auto visitTupleExcept(V &&Visitor, T &Tuple, E *Exclude) {
  auto WrappedVisitor = [&Visitor, Exclude](auto &Field) {
    if constexpr (std::is_same_v<std::decay_t<decltype(Field)>, E>) {
      // Make sure we don't visit the type system
      if (&Field != Exclude) {
        return Visitor(Field);
      }
    } else {
      return Visitor(Field);
    }
  };
  return visitTuple(WrappedVisitor, Tuple);
}

void model::purgeUnnamedAndUnreachableTypes(TupleTree<model::Binary> &Model) {
  purgeTypesImpl(Model, true);
}

void model::purgeUnreachableTypes(TupleTree<model::Binary> &Model) {
  purgeTypesImpl(Model, false);
}

static void model::purgeTypesImpl(TupleTree<model::Binary> &Model,
                                  bool KeepTypesWithName) {
  struct NodeData {
    model::TypeDefinition *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;

  Graph TypeGraph;
  std::map<const model::TypeDefinition *, Node *> TypeToNode;

  llvm::SmallPtrSet<model::TypeDefinition *, 16> ToKeep;

  // Remember those types we want to preserve.
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    if (KeepTypesWithName)
      if (not T->CustomName().empty() or not T->OriginalName().empty())
        ToKeep.insert(T.get());

    TypeToNode[T.get()] = TypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      TypeToNode.at(T.get())->addSuccessor(TypeToNode.at(DependantType));
    }
  }

  // Record references to types *outside* of Model->Types
  auto VisitBinary = [&](auto &Field) {
    auto Visitor = [&](auto &Element) {
      using type = std::decay_t<decltype(Element)>;
      if constexpr (std::is_same_v<type, DefinitionReference>)
        if (Element.isValid())
          ToKeep.insert(Element.get());
    };
    visitTupleTree(Field, Visitor, [](auto) {});
  };
  visitTupleExcept(VisitBinary, *Model, &Model->TypeDefinitions());

  // Visit all the nodes reachable from ToKeep
  df_iterator_default_set<Node *> Visited;
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions())
    if (isa<model::PrimitiveDefinition>(T.get()))
      Visited.insert(TypeToNode.at(T.get()));

  for (model::TypeDefinition *T : ToKeep)
    for (Node *N : depth_first_ext(TypeToNode.at(T), Visited))
      ;

  // Purge the non-visited
  llvm::erase_if(Model->TypeDefinitions(),
                 [&](model::UpcastableTypeDefinition &P) {
                   return not Visited.contains(TypeToNode.at(P.get()));
                 });
}
