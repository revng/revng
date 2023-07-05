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

static RegisterModelPass RPruneUnusedTypes("prune-unused-types",
                                           "Remove all the types that cannot "
                                           "be reached from "
                                           "any type from a Function ",
                                           model::pruneUnusedTypes);

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
  purgeTypesImpl<false>(Model);
}

void model::pruneUnusedTypes(TupleTree<model::Binary> &Model) {
  purgeTypesImpl<true>(Model);
}

template<bool PruneAllUnusedTypes>
void model::purgeTypesImpl(TupleTree<model::Binary> &Model) {
  struct NodeData {
    model::Type *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;

  Graph TypeGraph;
  std::map<const model::Type *, Node *> TypeToNode;

  llvm::SmallPtrSet<Type *, 16> ToKeep;

  // Remember those types we want to preserve.
  if constexpr (PruneAllUnusedTypes) {
    for (const auto &Function : Model->Functions()) {
      if (Function.Prototype().isValid()) {
        ToKeep.insert(const_cast<Type *>(Function.Prototype().get()));
      }
    }

    for (UpcastablePointer<model::Type> &T : Model->Types()) {
      TypeToNode[T.get()] = TypeGraph.addNode(NodeData{ T.get() });
    }
  } else {
    for (UpcastablePointer<model::Type> &T : Model->Types()) {

      if (not T->CustomName().empty() or not T->OriginalName().empty())
        ToKeep.insert(T.get());

      TypeToNode[T.get()] = TypeGraph.addNode(NodeData{ T.get() });
    }
  }

  // Create type system edges
  for (UpcastablePointer<model::Type> &T : Model->Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      TypeToNode.at(T.get())->addSuccessor(TypeToNode.at(DependantType));
    }
  }

  // Record references to types *outside* of Model->Types
  if constexpr (!PruneAllUnusedTypes) {
    auto VisitBinary = [&](auto &Field) {
      auto Visitor = [&](auto &Element) {
        using type = std::decay_t<decltype(Element)>;
        if constexpr (std::is_same_v<type, TypePath>)
          if (Element.isValid())
            ToKeep.insert(Element.get());
      };
      visitTupleTree(Field, Visitor, [](auto) {});
    };
    visitTupleExcept(VisitBinary, *Model, &Model->Types());
  }

  // Visit all the nodes reachable from ToKeep
  df_iterator_default_set<Node *> Visited;
  for (const UpcastablePointer<Type> &T : Model->Types())
    if (isa<model::PrimitiveType>(T.get()))
      Visited.insert(TypeToNode.at(T.get()));

  for (Type *T : ToKeep)
    for (Node *N : depth_first_ext(TypeToNode.at(T), Visited))
      ;

  // Purge the non-visited
  llvm::erase_if(Model->Types(), [&](UpcastablePointer<model::Type> &P) {
    return not Visited.contains(TypeToNode.at(P.get()));
  });
}
