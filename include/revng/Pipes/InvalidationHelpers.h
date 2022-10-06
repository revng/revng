#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringRef.h"

#include "revng/ABI/FunctionType.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/FunctionTags.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"

namespace revng::kinds {

namespace detail {

namespace CR = yield::crossrelations;

} // namespace detail

// Find callers of a given `model::Function` via CrossRelations
inline void insertCallers(const TupleTree<detail::CR::CrossRelations> &CR,
                          MetaAddress Entry,
                          std::set<MetaAddress> &Result) {
  const auto Location = location(revng::ranks::Function, Entry);
  const auto &Related = CR->Relations().at(Location.toString()).Related();
  for (const detail::CR::RelationTarget &RT : Related) {
    if (RT.Kind() == detail::CR::RelationType::IsCalledFrom) {
      auto Caller = genericLocationFromString<0>(RT.Location(),
                                                 revng::ranks::Function,
                                                 revng::ranks::BasicBlock,
                                                 revng::ranks::Instruction);
      revng_assert(Caller.has_value());
      Result.insert(std::get<0>(*Caller));
    }
  }
}

// Find callers of a given `model::DynamicFunction` via CrossRelations
inline void insertCallers(const TupleTree<detail::CR::CrossRelations> &CR,
                          std::string Entry,
                          std::set<MetaAddress> &Result) {
  const auto Location = location(revng::ranks::DynamicFunction, Entry);
  const auto &Related = CR->Relations().at(Location.toString()).Related();
  for (const detail::CR::RelationTarget &RT : Related) {
    if (RT.Kind() == detail::CR::RelationType::IsDynamicallyCalledFrom) {
      auto Caller = genericLocationFromString<0>(RT.Location(),
                                                 revng::ranks::Function,
                                                 revng::ranks::BasicBlock,
                                                 revng::ranks::Instruction);
      revng_assert(Caller.has_value());
      Result.insert(std::get<0>(*Caller));
    }
  }
}

// Invalidate all the targets belonging to function rank
inline void
invalidateAllTargetsPerFunctionRank(const TupleTree<model::Binary> &Model,
                                    std::set<MetaAddress> &Result) {
  for (const model::Function &Function : Model->Functions())
    Result.insert(Function.Entry());
}

namespace detail {

using CrossRelationTree = TupleTree<CR::CrossRelations>;

inline void
insertCallersAndTransitiveClosureInternal(const auto &CRGraph,
                                          const TupleTree<model::Binary> &Model,
                                          const std::string &Location,
                                          std::set<MetaAddress> &Result) {
  namespace ranks = revng::ranks;
  using CRNode = typename std::remove_cvref_t<decltype(CRGraph)>::Node;

  const CRNode *EntryNode = nullptr;
  for (const CRNode *Node : CRGraph.nodes())
    if (Node->data() == Location)
      EntryNode = Node;

  revng_assert(EntryNode != nullptr);

  // Inverse DFS visit that aims at finding its callers, and its
  // transitive closure, if they are all `Inline`.
  llvm::df_iterator_default_set<const CRNode *> NoInlineFunctions;
  for (const CRNode *Node : CRGraph.nodes()) {
    auto MaybeKey = pipeline::genericLocationFromString<0>(Node->data(),
                                                           ranks::Function,
                                                           ranks::BasicBlock,
                                                           ranks::Instruction);
    if (MaybeKey.has_value()) {
      auto It = Model->Functions().find(std::get<MetaAddress>(*MaybeKey));
      if (It != Model->Functions().end())
        if (It->Attributes().count(model::FunctionAttribute::Inline) == 0)
          NoInlineFunctions.insert(Node);
    }
  }

  llvm::df_iterator_default_set<const CRNode *> AdditionalSet;
  auto VisitRange = llvm::inverse_post_order_ext(EntryNode, AdditionalSet);

  for (const CRNode *Node : VisitRange) {
    if (!llvm::is_contained(NoInlineFunctions, Node)) {
      AdditionalSet.insert(Node);
      continue;
    }

    auto Address = pipeline::genericLocationFromString<0>(Node->data(),
                                                          ranks::Function,
                                                          ranks::BasicBlock,
                                                          ranks::Instruction);
    if (Address.has_value())
      Result.insert(std::get<MetaAddress>(*Address));
  }
}

} // namespace detail

inline void
insertCallersAndTransitiveClosureIfInline(const auto &CRGraph,
                                          const TupleTree<model::Binary> &Model,
                                          MetaAddress Entry,
                                          std::set<MetaAddress> &Result) {
  const auto Location = serializedLocation(revng::ranks::Function, Entry);
  detail::insertCallersAndTransitiveClosureInternal(CRGraph,
                                                    Model,
                                                    Location,
                                                    Result);
}

inline void
insertCallersAndTransitiveClosureIfInline(const auto &CRGraph,
                                          const TupleTree<model::Binary> &Model,
                                          std::string Entry,
                                          std::set<MetaAddress> &Result) {
  const auto Location = serializedLocation(revng::ranks::DynamicFunction,
                                           Entry);
  detail::insertCallersAndTransitiveClosureInternal(CRGraph,
                                                    Model,
                                                    Location,
                                                    Result);
}

} // namespace revng::kinds
