/// \file CallGraph.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/ADT/STLExtras.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"

using CR = yield::CrossRelations;
CR::CrossRelations(const SortedVector<efa::FunctionMetadata> &Metadata,
                   const model::Binary &Binary) {
  revng_assert(Metadata.size() == Binary.Functions().size());

  namespace ranks = revng::ranks;

  // Make sure all the functions are present.
  for (auto Inserter = Relations().batch_insert();
       const auto &Function : Binary.Functions()) {
    const auto Location = pipeline::location(ranks::Function, Function.Entry());
    Inserter.insert(yield::RelationDescription(Location.toString(), {}));
  }

  // Make sure all the dynamic functions are present
  for (auto Inserter = Relations.batch_insert();
       const auto &Function : Binary.ImportedDynamicFunctions()) {
    const auto Location = pipeline::location(ranks::DynamicFunction,
                                             Function.OriginalName());
    Inserter.insert(yield::RelationDescription(Location.toString(), {}));
  }

  for (const auto &[EntryAddress, ControlFlowGraph] : Metadata) {
    for (const auto &BasicBlock : ControlFlowGraph) {
      for (const auto &Edge : BasicBlock.Successors()) {
        if (auto *CallEdge = llvm::dyn_cast<yield::CallEdge>(Edge.get())) {
          if (efa::FunctionEdgeType::isCall(Edge->Type())) {
            if (const auto &Callee = Edge->Destination(); Callee.isValid()) {
              // TODO: embed information about the call instruction into
              //       `CallLocation` after yield starts providing it.

              auto L = pipeline::location(ranks::Function, Callee).toString();
              if (auto It = Relations().find(L); It != Relations().end()) {
                yield::RelationTarget T(yield::RelationType::IsCalledFrom,
                                        CallLocation.toString());
                It->Related().insert(std::move(T));
              }
            } else if (!CallEdge->DynamicFunction.empty()) {
              auto L = pipeline::location(ranks::DynamicFunction,
                                          CallEdge->DynamicFunction)
                         .toString();
              if (auto It = Relations().find(L); It != Relations().end()) {
                using namespace yield::RelationType;
                yield::RelationTarget T(IsDynamicallyCalledFrom,
                                        CallLocation.toString());
                It->Related().insert(std::move(T));
              }
            } else {
              // Ignore indirect calls.
            }
          } else {
            // Ignore non-call edges.
          }
        }
      }
    }
  }
}

template<typename AddNodeCallable, typename AddEdgeCallable>
static void conversionHelper(const yield::CrossRelations &Input,
                             const AddNodeCallable &AddNode,
                             const AddEdgeCallable &AddEdge) {
  for (const auto &[LocationString, Related] : Input.Relations())
    AddNode(LocationString);

  for (const auto &[LocationString, Related] : Input.Relations()) {
    for (const auto &[RelationKind, TargetString] : Related) {
      switch (RelationKind) {
      case yield::RelationType::IsCalledFrom:
      case yield::RelationType::IsDynamicallyCalledFrom:
        AddEdge(LocationString, TargetString, RelationKind);
        break;

      case yield::RelationType::Invalid:
      case yield::RelationType::Count:
      default:
        revng_abort("Unknown enum value");
      }
    }
  }
}

GenericGraph<yield::CrossRelations::Node, 16, true>
yield::CrossRelations::toCallGraph() const {
  GenericGraph<yield::CrossRelations::Node, 16, true> Result;

  using NodeView = decltype(Result)::Node *;
  std::unordered_map<std::string_view, NodeView> LookupHelper;
  auto AddNode = [&Result, &LookupHelper](std::string_view Location) {
    auto *Node = Result.addNode(Location);
    auto [Iterator, Success] = LookupHelper.try_emplace(Location, Node);
    revng_assert(Success);
  };
  auto AddEdge = [&LookupHelper](std::string_view Callee,
                                 std::string_view Caller,
                                 yield::RelationType::Values Kind) {
    if (Kind == yield::RelationType::IsCalledFrom
        || Kind == yield::RelationType::IsDynamicallyCalledFrom) {
      // This assumes all the call sites are represented as basic block
      // locations for all the relations covered by these two kinds.
      using namespace pipeline;
      namespace ranks = revng::ranks;
      auto CallerLocation = *locationFromString(ranks::BasicBlock, Caller);
      auto CallerFunction = convertLocation(ranks::Function, CallerLocation);
      auto *CallerNode = LookupHelper.at(CallerFunction.toString());
      auto *CalleeNode = LookupHelper.at(Callee);

      using EL = yield::CrossRelations::EdgeLabel;
      if (!llvm::is_contained(CallerNode->successors(), CalleeNode))
        CallerNode->addSuccessor(CalleeNode, EL{ Kind });
    } else {
      revng_abort("Unsupported relation type.");
    }
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}

yield::Graph yield::CrossRelations::toYieldGraph() const {
  yield::Graph Result;

  std::map<MetaAddress, yield::Graph::Node *> LookupHelper;

  namespace ranks = revng::ranks;
  namespace p = pipeline;
  auto AddNode = [&Result, &LookupHelper](std::string_view Location) {
    auto MaybeKey = p::genericLocationFromString<0>(Location,
                                                    ranks::Function,
                                                    ranks::BasicBlock,
                                                    ranks::Instruction);
    // TODO: extend to support dynamic functions
    if (MaybeKey.has_value()) {
      auto Address = std::get<0>(MaybeKey.value());
      auto [_, Success] = LookupHelper.try_emplace(Address,
                                                   Result.addNode(Address));
      revng_assert(Success);
    }
  };
  auto AddEdge = [&LookupHelper](std::string_view FromLocation,
                                 std::string_view ToLocation,
                                 yield::RelationType::Values Kind) {
    if (Kind == yield::RelationType::IsCalledFrom) {
      auto FromKey = p::genericLocationFromString<0>(FromLocation,
                                                     ranks::Function,
                                                     ranks::BasicBlock,
                                                     ranks::Instruction);
      auto ToKey = p::genericLocationFromString<0>(ToLocation,
                                                   ranks::Function,
                                                   ranks::BasicBlock,
                                                   ranks::Instruction);
      revng_assert(FromKey.has_value() && ToKey.has_value());

      auto *FromNode = LookupHelper.at(std::get<0>(FromKey.value()));
      auto *ToNode = LookupHelper.at(std::get<0>(ToKey.value()));
      if (!llvm::is_contained(FromNode->successors(), ToNode))
        FromNode->addSuccessor(ToNode);
    }
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}
