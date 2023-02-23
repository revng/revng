/// \file Extraction.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeType.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Extraction.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Graph.h"

yield::Graph
yield::cfg::extractFromInternal(const yield::Function &Function,
                                const model::Binary &Binary,
                                const Configuration &Configuration) {
  const auto &ControlFlowGraph = Function.ControlFlowGraph();
  auto [Result, Table] = efa::buildControlFlowGraph<Graph>(ControlFlowGraph,
                                                           Function.Entry(),
                                                           Binary);

  if (!Configuration.AddExitNode) {
    auto ExitNodeIterator = Table.find(BasicBlockID::invalid());
    if (ExitNodeIterator != Table.end())
      Result.removeNode(ExitNodeIterator->second);
  }

  if (Configuration.AddEntryNode) {
    auto EntryIterator = Table.find(BasicBlockID(Function.Entry()));
    revng_assert(EntryIterator != Table.end());
    auto *RootNode = Result.addNode();
    RootNode->Address = BasicBlockID::invalid();
    RootNode->addSuccessor(EntryIterator->second);
    Result.setEntryNode(RootNode);
  }

  // Colour taken and refused edges.
  for (const auto &BasicBlock : Function.ControlFlowGraph()) {
    auto NodeIterator = Table.find(BasicBlock.ID());
    revng_assert(NodeIterator != Table.end());
    auto &CurrentNode = *NodeIterator->second;
    CurrentNode.NextAddress = BasicBlock.nextBlock();

    if (BasicBlock.Successors().size() == 2) {
      revng_assert(CurrentNode.successorCount() <= 2);
      if (CurrentNode.successorCount() == 2) {
        auto Front = *CurrentNode.successor_edges_begin();
        auto Back = *std::next(CurrentNode.successor_edges_begin());
        if (Front.Neighbor->Address == BasicBlock.nextBlock()) {
          Front.Label->Type = yield::Graph::EdgeType::Refused;
          Back.Label->Type = yield::Graph::EdgeType::Taken;
        } else if (Back.Neighbor->Address == BasicBlock.nextBlock()) {
          Front.Label->Type = yield::Graph::EdgeType::Taken;
          Back.Label->Type = yield::Graph::EdgeType::Refused;
        }
      } else if (CurrentNode.successorCount() == 1) {
        for (const auto &Successor : BasicBlock.Successors())
          if (FunctionEdgeType::isCall(Successor->Type()))
            continue;

        auto Edge = *CurrentNode.successor_edges_begin();
        if (Edge.Neighbor->Address == BasicBlock.nextBlock())
          Edge.Label->Type = yield::Graph::EdgeType::Refused;
      }
    }
  }

  return std::move(Result);
}
