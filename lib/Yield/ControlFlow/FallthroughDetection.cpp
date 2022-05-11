/// \file FallthroughDetection.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"

template<bool ShouldMergeFallthroughTargets>
const yield::BasicBlock *
yield::cfg::detectFallthrough(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  const yield::BasicBlock *Result = nullptr;

  for (const auto &Edge : BasicBlock.Successors) {
    auto [NextAddress, _] = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (NextAddress.isValid() && NextAddress == BasicBlock.End) {
      if (auto Iterator = Function.ControlFlowGraph.find(NextAddress);
          Iterator != Function.ControlFlowGraph.end()) {
        if constexpr (ShouldMergeFallthroughTargets) {
          if (Iterator->IsLabelAlwaysRequired == false) {
            revng_assert(Result == nullptr,
                         "Multiple targets with the same address");
            Result = &*Iterator;
          }
        }
      }
    }
  }

  return Result;
}

template const yield::BasicBlock *
yield::cfg::detectFallthrough<true>(const yield::BasicBlock &BasicBlock,
                                    const yield::Function &Function,
                                    const model::Binary &Binary);
template const yield::BasicBlock *
yield::cfg::detectFallthrough<false>(const yield::BasicBlock &BasicBlock,
                                     const yield::Function &Function,
                                     const model::Binary &Binary);
