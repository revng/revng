/// \file FallthroughDetection.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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

  for (const MetaAddress &Target : BasicBlock.Targets) {
    if (Target.isValid() && Target == BasicBlock.NextAddress) {
      auto Iterator = Function.BasicBlocks.find(Target);
      if (Iterator != Function.BasicBlocks.end()) {
        using namespace yield::BasicBlockType;
        if (shouldSkip<ShouldMergeFallthroughTargets>(Iterator->Type)) {
          revng_assert(Result == nullptr,
                       "Multiple targets with the same address");
          Result = &*Iterator;
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
