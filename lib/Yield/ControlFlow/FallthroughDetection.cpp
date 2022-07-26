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

const yield::BasicBlock *
yield::cfg::detectFallthrough(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  const yield::BasicBlock *Result = nullptr;

  for (const auto &Edge : BasicBlock.Successors) {
    auto [Next, _] = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (Next.has_value() && Next->isValid() && Next == BasicBlock.End) {
      if (auto Iterator = Function.ControlFlowGraph.find(Next.value());
          Iterator != Function.ControlFlowGraph.end()) {
        if (Iterator->IsLabelAlwaysRequired == false) {
          revng_assert(Result == nullptr,
                       "Multiple targets with the same address");
          Result = &*Iterator;
        }
      }
    }
  }

  return Result;
}

llvm::SmallVector<const yield::BasicBlock *, 8>
yield::cfg::labeledBlock(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary) {
  // Blocks that are a part of another labeled block cannot start a new one.
  if (BasicBlock.IsLabelAlwaysRequired == false)
    return {};

  llvm::SmallVector<const yield::BasicBlock *, 8> Result = { &BasicBlock };

  auto Next = detectFallthrough(BasicBlock, Function, Binary);
  while (Next != nullptr)
    Next = detectFallthrough(*Result.emplace_back(Next), Function, Binary);

  return Result;
}
