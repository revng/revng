/// \file Extraction.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CFGHelpers.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeType.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Extraction.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/ControlFlow/Graph.h"
#include "revng/Yield/Function.h"

yield::cfg::PreLayoutGraph
yield::cfg::extractFromInternal(const yield::Function &Function,
                                const model::Binary &Binary,
                                const Configuration &Configuration) {
  using PLG = PreLayoutGraph;
  const auto &ControlFlowGraph = Function.Blocks();
  auto [Result, Lookup] = efa::buildControlFlowGraph<PLG>(ControlFlowGraph,
                                                          Function.Entry(),
                                                          Binary);

  if (!Configuration.AddExitNode) {
    auto ExitNodeIterator = Lookup.find(BasicBlockID::invalid());
    if (ExitNodeIterator != Lookup.end())
      Result.removeNode(ExitNodeIterator->second);
  }

  if (Configuration.AddEntryNode and ControlFlowGraph.size() != 0) {
    auto EntryIterator = Lookup.find(BasicBlockID(Function.Entry()));
    revng_assert(EntryIterator != Lookup.end());
    auto *RootNode = Result.addNode();
    RootNode->addSuccessor(EntryIterator->second);
    Result.setEntryNode(RootNode);
  }

  // Colour 'taken' and 'refused' edges.
  for (const auto &BasicBlock : Function.Blocks()) {
    auto NodeIterator = Lookup.find(BasicBlock.ID());
    revng_assert(NodeIterator != Lookup.end());
    auto &CurrentNode = *NodeIterator->second;

    if (BasicBlock.Successors().size() == 2) {
      revng_assert(CurrentNode.successorCount() <= 2);
      if (CurrentNode.successorCount() == 2) {
        auto Front = *CurrentNode.successor_edges_begin();
        auto Back = *std::next(CurrentNode.successor_edges_begin());
        if (Front.Neighbor->getBasicBlock() == BasicBlock.nextBlock()) {
          Front.Label->Type = yield::cfg::EdgeType::Refused;
          Back.Label->Type = yield::cfg::EdgeType::Taken;
        } else if (Back.Neighbor->getBasicBlock() == BasicBlock.nextBlock()) {
          Front.Label->Type = yield::cfg::EdgeType::Taken;
          Back.Label->Type = yield::cfg::EdgeType::Refused;
        }
      } else if (CurrentNode.successorCount() == 1) {
        for (const auto &Successor : BasicBlock.Successors())
          if (FunctionEdgeType::isCall(Successor->Type()))
            continue;

        auto Edge = *CurrentNode.successor_edges_begin();
        if (Edge.Neighbor->getBasicBlock() == BasicBlock.nextBlock())
          Edge.Label->Type = yield::cfg::EdgeType::Refused;
      }
    }
  }

  return std::move(Result);
}
