/// \file CallGraph.cpp

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
  using pipeline::serializedLocation;

  // Make sure all the functions are present.
  for (auto Inserter = Relations().batch_insert();
       const auto &Function : Binary.Functions()) {
    auto Location = serializedLocation(ranks::Function, Function.key());
    Inserter.insert(CR::RelationDescription(std::move(Location), {}));
  }

  // Make sure all the dynamic functions are also present
  for (auto Inserter = Relations().batch_insert();
       const auto &Function : Binary.ImportedDynamicFunctions()) {
    auto Location = serializedLocation(ranks::DynamicFunction, Function.key());
    Inserter.insert(CR::RelationDescription(std::move(Location), {}));
  }

  for (const auto &[EntryAddress, _, ControlFlowGraph] : Metadata) {
    for (const auto &BasicBlock : ControlFlowGraph) {
      auto CallLocation = serializedLocation(ranks::BasicBlock,
                                             EntryAddress,
                                             BasicBlock.ID());

      for (const auto &Edge : BasicBlock.Successors()) {
        if (auto *CallEdge = llvm::dyn_cast<efa::CallEdge>(Edge.get())) {
          if (efa::FunctionEdgeType::isCall(Edge->Type())) {
            if (const auto &Callee = Edge->Destination(); Callee.isValid()) {
              // TODO: embed information about the call instruction into
              //       `CallLocation` after metadata starts providing it.
              const auto L = serializedLocation(ranks::Function,
                                                Callee.notInlinedAddress());
              if (auto It = Relations().find(L); It != Relations().end())
                It->IsCalledFrom().emplace(std::move(CallLocation));
            } else if (!CallEdge->DynamicFunction().empty()) {
              const auto L = serializedLocation(ranks::DynamicFunction,
                                                CallEdge->DynamicFunction());
              if (auto It = Relations().find(L); It != Relations().end())
                It->IsCalledFrom().emplace(std::move(CallLocation));
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

  for (const CR::RelationDescription &Relation : Input.Relations())
    for (std::string_view CallerLocation : Relation.IsCalledFrom())
      AddEdge(Relation.Location(), CallerLocation);
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
                                 std::string_view Caller) {
    // This assumes all the call sites are represented as basic block
    // locations for all the relations covered by these two kinds.
    using namespace pipeline;
    namespace ranks = revng::ranks;
    auto CallerLocation = *locationFromString(ranks::BasicBlock, Caller);
    auto CallerFunction = convertLocation(ranks::Function, CallerLocation);
    auto *CallerNode = LookupHelper.at(CallerFunction.toString());
    auto *CalleeNode = LookupHelper.at(Callee);

    if (!llvm::is_contained(CallerNode->successors(), CalleeNode))
      CallerNode->addSuccessor(CalleeNode);
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}

yield::calls::PreLayoutGraph CR::CrossRelations::toYieldGraph() const {
  yield::calls::PreLayoutGraph Result;

  using GraphNode = yield::calls::PreLayoutGraph::Node;
  std::unordered_map<std::string_view, GraphNode *> LookupHelper;

  namespace ranks = revng::ranks;
  using pipeline::locationFromString;
  auto AddNode = [&Result, &LookupHelper](std::string_view Location) {
    GraphNode *Node = nullptr;
    if (auto Dynamic = locationFromString(ranks::DynamicFunction, Location))
      Node = Result.addNode(*Dynamic);
    else if (auto Function = locationFromString(ranks::Function, Location))
      Node = Result.addNode(*Function);
    else
      revng_abort("Unsupported location found in cross relations.");

    auto [Iterator, Success] = LookupHelper.try_emplace(Location, Node);
    revng_assert(Success);
  };
  auto AddEdge = [&LookupHelper](std::string_view Callee,
                                 std::string_view Caller) {
    // This assumes all the call sites are represented as basic block
    // locations for all the relations covered by these two kinds.
    auto CallerLocation = *locationFromString(ranks::BasicBlock, Caller);
    auto CallerFunction = convertLocation(ranks::Function, CallerLocation);
    auto *CallerNode = LookupHelper.at(CallerFunction.toString());
    auto *CalleeNode = LookupHelper.at(Callee);

    if (!llvm::is_contained(CallerNode->successors(), CalleeNode))
      CallerNode->addSuccessor(CalleeNode);
  };
  conversionHelper(*this, AddNode, AddEdge);

  return Result;
}
