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
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"

namespace CR = yield::crossrelations;

using MetadataContainer = SortedVector<efa::FunctionMetadata>;
CR::CrossRelations::CrossRelations(const MetadataContainer &Metadata,
                                   const model::Binary &Binary) {
  revng_assert(Metadata.size() == Binary.Functions().size());

  namespace ranks = revng::ranks;

  // Make sure all the functions are present.
  for (auto Inserter = Relations().batch_insert();
       const auto &Function : Binary.Functions()) {
    const auto Location = pipeline::location(ranks::Function, Function.Entry());
    Inserter.insert(CR::RelationDescription(Location.toString(), {}));
  }

  // Make sure all the dynamic functions are also present
  for (auto Inserter = Relations().batch_insert();
       const auto &Function : Binary.ImportedDynamicFunctions()) {
    const auto Location = pipeline::location(ranks::DynamicFunction,
                                             Function.OriginalName());
    Inserter.insert(CR::RelationDescription(Location.toString(), {}));
  }

  for (const auto &[EntryAddress, ControlFlowGraph] : Metadata) {
    for (const auto &BasicBlock : ControlFlowGraph) {
      auto CallLocation = pipeline::location(ranks::BasicBlock,
                                             EntryAddress,
                                             BasicBlock.ID());

      for (const auto &Edge : BasicBlock.Successors()) {
        if (auto *CallEdge = llvm::dyn_cast<efa::CallEdge>(Edge.get())) {
          if (efa::FunctionEdgeType::isCall(Edge->Type())) {
            if (const auto &Callee = Edge->Destination(); Callee.isValid()) {
              // TODO: embed information about the call instruction into
              //       `CallLocation` after yield starts providing it.

              auto CalleeAddress = Callee.notInlinedAddress();
              auto L = pipeline::location(ranks::Function, CalleeAddress)
                         .toString();
              if (auto It = Relations().find(L); It != Relations().end()) {
                CR::RelationTarget T(CR::RelationType::IsCalledFrom,
                                     CallLocation.toString());
                It->Related().insert(std::move(T));
              }
            } else if (!CallEdge->DynamicFunction().empty()) {
              auto L = pipeline::location(ranks::DynamicFunction,
                                          CallEdge->DynamicFunction())
                         .toString();
              if (auto It = Relations().find(L); It != Relations().end()) {
                CR::RelationTarget T(CR::RelationType::IsDynamicallyCalledFrom,
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
static void conversionHelper(const CR::CrossRelations &Input,
                             const AddNodeCallable &AddNode,
                             const AddEdgeCallable &AddEdge) {
  for (const auto &[LocationString, Related] : Input.Relations())
    AddNode(LocationString);

  for (const auto &[LocationString, Related] : Input.Relations()) {
    for (const auto &[RelationKind, TargetString] : Related) {
      switch (RelationKind) {
      case CR::RelationType::IsCalledFrom:
      case CR::RelationType::IsDynamicallyCalledFrom:
        AddEdge(LocationString, TargetString, RelationKind);
        break;

      case CR::RelationType::Invalid:
      case CR::RelationType::Count:
      default:
        revng_abort("Unknown enum value");
      }
    }
  }
}

GenericGraph<CR::CrossRelations::Node, 16, true>
CR::CrossRelations::toCallGraph() const {
  GenericGraph<CR::CrossRelations::Node, 16, true> Result;

  using NodeView = decltype(Result)::Node *;
  std::unordered_map<std::string_view, NodeView> LookupHelper;
  auto AddNode = [&Result, &LookupHelper](std::string_view Location) {
    auto *Node = Result.addNode(Location);
    auto [Iterator, Success] = LookupHelper.try_emplace(Location, Node);
    revng_assert(Success);
  };
  auto AddEdge = [&LookupHelper](std::string_view Callee,
                                 std::string_view Caller,
                                 RelationType::Values Kind) {
    if (Kind == CR::RelationType::IsCalledFrom
        || Kind == CR::RelationType::IsDynamicallyCalledFrom) {
      // This assumes all the call sites are represented as basic block
      // locations for all the relations covered by these two kinds.
      using namespace pipeline;
      namespace ranks = revng::ranks;
      auto CallerLocation = *locationFromString(ranks::BasicBlock, Caller);
      auto CallerFunction = convertLocation(ranks::Function, CallerLocation);
      auto *CallerNode = LookupHelper.at(CallerFunction.toString());
      auto *CalleeNode = LookupHelper.at(Callee);

      using EdgeLabel = CR::CrossRelations::EdgeLabel;
      if (!llvm::is_contained(CallerNode->successors(), CalleeNode))
        CallerNode->addSuccessor(CalleeNode, EdgeLabel{ Kind });
    } else {
      revng_abort("Unsupported relation type.");
    }
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}

yield::Graph CR::CrossRelations::toYieldGraph() const {
  yield::Graph Result;

  std::map<BasicBlockID, yield::Graph::Node *> LookupHelper;

  namespace ranks = revng::ranks;
  using namespace pipeline;

  auto AddNode = [&Result, &LookupHelper](std::string_view Location) {
    auto MaybeKey = genericLocationFromString<0>(Location,
                                                 ranks::Function,
                                                 ranks::BasicBlock,
                                                 ranks::Instruction);
    // TODO: extend to support dynamic functions
    if (MaybeKey.has_value()) {
      auto Address = BasicBlockID(std::get<0>(MaybeKey.value()));
      auto [_, Success] = LookupHelper.try_emplace(Address,
                                                   Result.addNode(Address));
      revng_assert(Success);
    }
  };
  auto AddEdge = [&LookupHelper](std::string_view FromLocation,
                                 std::string_view ToLocation,
                                 CR::RelationType::Values Kind) {
    if (Kind == CR::RelationType::IsCalledFrom) {
      using namespace pipeline;
      auto GetBlockID = [](llvm::StringRef Location) -> BasicBlockID {
        auto MaybeAddress = genericLocationFromString<0>(Location,
                                                         ranks::Function,
                                                         ranks::BasicBlock,
                                                         ranks::Instruction);
        revng_assert(MaybeAddress);
        return BasicBlockID(std::get<0>(*MaybeAddress));
      };

      auto *FromNode = LookupHelper.at(GetBlockID(FromLocation));
      auto *ToNode = LookupHelper.at(GetBlockID(ToLocation));
      if (!llvm::is_contained(FromNode->successors(), ToNode))
        FromNode->addSuccessor(ToNode);
    }
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}
