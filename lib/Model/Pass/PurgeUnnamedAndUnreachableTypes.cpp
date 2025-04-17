/// \file PurgeUnnamedAndUnreachableTypes.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Filters.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Model/VerifyHelper.h"

using namespace llvm;

static Logger<> Log("purge-model-types");

static RegisterModelPass R0("purge-invalid-types",
                            "Remove all the types that do not verify",
                            model::purgeInvalidTypes);

static RegisterModelPass R("purge-unnamed-and-unreachable-types",
                           "Remove all the types that cannot be reached from "
                           "any named type or a type \"outside\" the type "
                           "system itself",
                           model::purgeUnnamedAndUnreachableTypes);

static RegisterModelPass R2("purge-unreachable-types",
                            "Remove all the types that cannot be reached from "
                            "a type \"outside\" the type system itself",
                            model::purgeUnreachableTypes);

// Helper for fixing array constness.
static RecursiveCoroutine<void> fixConstArrays(model::Type &Type) {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    if (Array->IsConst()) {
      Array->ElementType()->IsConst() = true;
      Array->IsConst() = false;
    }

    if (not Array->ElementType().isEmpty())
      rc_recur fixConstArrays(*Array->ElementType());

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    if (not Pointer->PointeeType().isEmpty())
      rc_recur fixConstArrays(*Pointer->PointeeType());

  } else {
    // Here we assume that this helper is run for every definition, as such
    // there's no need to unwrap `model::DefinedType`s.
    revng_assert(llvm::isa<model::DefinedType>(Type)
                 || llvm::isa<model::PrimitiveType>(Type));
  }
}

void model::purgeInvalidTypes(TupleTree<model::Binary> &Model) {
  model::VerifyHelper VH;

  // Ensure there are no const arrays, since we explicitly disallow those.
  for (auto &Definition : Model->TypeDefinitions())
    for (model::Type *Edge : Definition->edges())
      fixConstArrays(*Edge);

  // To avoid removing entire structs/unions, remove invalid fields first.
  auto IsFieldInvalid = [&VH](const auto &Field) {
    std::optional<uint64_t> MaybeSize = Field.Type()->size(VH);
    return !Field.Type()->verify(VH) && MaybeSize.has_value();
  };
  for (auto &&Struct : Model->TypeDefinitions() | model::filter::Struct)
    Struct.Fields().erase_if(IsFieldInvalid);
  for (auto &&Union : Model->TypeDefinitions() | model::filter::Union) {
    size_t ErasedCount = Union.Fields().erase_if(IsFieldInvalid);

    if (ErasedCount != 0) {
      // We have removed some entries, so we need to update their indices.
      //
      // Note: changing the key of an element in a sorted container shouldn't
      // be allowed. However, it should be fine in this case, since we
      // preserve the ordering.
      for (auto &[Index, Field] : llvm::enumerate(Union.Fields()))
        Field.Index() = Index;
      revng_assert(Union.Fields().isSorted());
    }
  }

  // If there are still any invalid types, we have to get rid of them.
  auto IsTypeInvalid = [&VH](const model::UpcastableTypeDefinition &T) {
    return !T->verify(VH);
  };
  auto ToDrop = Model->TypeDefinitions() | std::views::filter(IsTypeInvalid)
                | std::views::transform([](const auto &T) { return T.get(); })
                | revng::to<std::set<const model::TypeDefinition *>>();
  unsigned DroppedCount = dropTypesDependingOnDefinitions(Model, ToDrop);
  revng_log(Log, "Purging " << DroppedCount << " types.");
}

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
    const model::TypeDefinition *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;

  Graph TypeGraph;
  std::map<const model::TypeDefinition *, Node *> TypeToNode;

  llvm::SmallPtrSet<const model::TypeDefinition *, 16> ToKeep;

  // Remember those types we want to preserve.
  for (const model::UpcastableTypeDefinition &T : Model->TypeDefinitions()) {
    if (KeepTypesWithName)
      if (not T->Name().empty())
        ToKeep.insert(T.get());

    TypeToNode[T.get()] = TypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges
  for (const model::UpcastableTypeDefinition &D : Model->TypeDefinitions())
    for (const model::Type *Edge : D->edges())
      if (const model::TypeDefinition *Definition = Edge->skipToDefinition())
        TypeToNode.at(D.get())->addSuccessor(TypeToNode.at(Definition));

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
  for (const model::TypeDefinition *T : ToKeep)
    for (Node *N : depth_first_ext(TypeToNode.at(T), Visited))
      ;

  // Purge the non-visited
  llvm::erase_if(Model->TypeDefinitions(),
                 [&](const model::UpcastableTypeDefinition &P) {
                   return not Visited.contains(TypeToNode.at(P.get()));
                 });
}
