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
#include "revng/Yield/CrossRelations.h"

using yield::CrossRelations;

namespace revng::kinds {

// Find callers of a given `model::Function` via CrossRelations
inline void insertCallers(const TupleTree<CrossRelations> &CR,
                          MetaAddress Entry,
                          std::set<MetaAddress> &Result) {
  const auto Location = location(revng::ranks::Function, Entry);
  const auto &Related = CR->Relations.at(Location.toString()).Related;
  for (const yield::RelationTarget &RT : Related) {
    if (RT.Kind == yield::RelationType::IsCalledFrom) {
      auto Caller = genericLocationFromString<0>(RT.Location,
                                                 revng::ranks::Function,
                                                 revng::ranks::BasicBlock,
                                                 revng::ranks::Instruction);
      revng_assert(Caller.has_value());
      Result.insert(std::get<0>(*Caller));
    }
  }
}

// Find callers of a given `model::DynamicFunction` via CrossRelations
inline void insertCallers(const TupleTree<CrossRelations> &CR,
                          std::string Entry,
                          std::set<MetaAddress> &Result) {
  const auto Location = location(revng::ranks::DynamicFunction, Entry);
  const auto &Related = CR->Relations.at(Location.toString()).Related;
  for (const yield::RelationTarget &RT : Related) {
    if (RT.Kind == yield::RelationType::IsDynamicallyCalledFrom) {
      auto Caller = genericLocationFromString<0>(RT.Location,
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
  for (const model::Function &Function : Model->Functions)
    Result.insert(Function.Entry);
}

namespace detail {

inline void
insertCallersAndTransitiveClosureInternal(const TupleTree<CrossRelations> &CR,
                                          const TupleTree<model::Binary> &Model,
                                          const std::string &Location,
                                          std::set<MetaAddress> &Result) {
  auto CRGraph = CR->toCallGraph();
  using CRNode = typename decltype(CRGraph)::Node;

  const CRNode *EntryNode = nullptr;
  for (const CRNode *Node : CRGraph.nodes())
    if (Node->data() == Location)
      EntryNode = Node;

  revng_assert(EntryNode != nullptr);

  // Inverse DFS visit that aims at finding its callers, and its
  // transitive closure, if they are all `Inline`.
  llvm::df_iterator_default_set<const CRNode *> NoInlineFunctions;
  auto VisitRange = llvm::inverse_post_order_ext(EntryNode, NoInlineFunctions);

  for (const CRNode *Node : VisitRange) {
    namespace p = pipeline;
    namespace ranks = revng::ranks;

    for (const CRNode *From : Node->predecessors()) {
      auto MaybeKey = p::genericLocationFromString<0>(From->data(),
                                                      ranks::Function,
                                                      ranks::BasicBlock,
                                                      ranks::Instruction);

      if (MaybeKey.has_value()) {
        auto &Function = Model->Functions.at(std::get<MetaAddress>(*MaybeKey));
        if (Function.Attributes.count(model::FunctionAttribute::Inline) == 0)
          NoInlineFunctions.insert(From);
      }
    }

    auto Address = p::genericLocationFromString<0>(Node->data(),
                                                   ranks::Function,
                                                   ranks::BasicBlock,
                                                   ranks::Instruction);
    if (Address.has_value())
      Result.insert(std::get<MetaAddress>(*Address));
  }
}

} // namespace detail

inline void
insertCallersAndTransitiveClosureIfInline(const TupleTree<CrossRelations> &CR,
                                          const TupleTree<model::Binary> &Model,
                                          MetaAddress Entry,
                                          std::set<MetaAddress> &Result) {
  const auto Location = serializedLocation(revng::ranks::Function, Entry);
  detail::insertCallersAndTransitiveClosureInternal(CR,
                                                    Model,
                                                    Location,
                                                    Result);
}

inline void
insertCallersAndTransitiveClosureIfInline(const TupleTree<CrossRelations> &CR,
                                          const TupleTree<model::Binary> &Model,
                                          std::string Entry,
                                          std::set<MetaAddress> &Result) {
  const auto Location = serializedLocation(revng::ranks::DynamicFunction,
                                           Entry);
  detail::insertCallersAndTransitiveClosureInternal(CR,
                                                    Model,
                                                    Location,
                                                    Result);
}

} // namespace revng::kinds
