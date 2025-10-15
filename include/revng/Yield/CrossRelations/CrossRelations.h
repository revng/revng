#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/CallGraphs/Graph.h"
#include "revng/Yield/CrossRelations/RelationDescription.h"

#include "revng/Yield/CrossRelations/Generated/Early/CrossRelations.h"

namespace yield::crossrelations {

class CrossRelations : public generated::CrossRelations {
private:
  using Node = BidirectionalNode<std::string>;

public:
  CrossRelations() = default;
  CrossRelations(const SortedVector<efa::ControlFlowGraph> &Metadata,
                 const model::Binary &Binary);

  GenericGraph<Node, 16, true> toCallGraph() const;
  yield::calls::PreLayoutGraph toYieldGraph() const;

  llvm::Error verify() const {
    if (Relations().empty())
      return revng::createError("Relocation map must not be empty.");

    for (const RelationDescription &Relation : Relations()) {
      using pipeline::locationFromString;
      bool IsCalleeValid = locationFromString(revng::ranks::DynamicFunction,
                                              Relation.Location())
                           || locationFromString(revng::ranks::Function,
                                                 Relation.Location());
      if (not IsCalleeValid)
        return revng::createError("Relocation map has an invalid callee: '"
                                  + Relation.Location() + "'.");

      for (std::string SerializedCaller : Relation.IsCalledFrom()) {
        auto IsCallerValid = locationFromString(revng::ranks::BasicBlock,
                                                SerializedCaller);
        if (not IsCallerValid)
          return revng::createError("Relocation map has an invalid caller: '"
                                    + SerializedCaller + "'.");
      }
    }

    return llvm::Error::success();
  }
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/CrossRelations.h"
