#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Support/IRHelpers.h"

// TODO: handle CABIFunctionType
inline model::RawFunctionType *
getCallSitePrototype(const model::Binary &Binary,
                     const model::Function &Function,
                     llvm::CallInst *Call) {
  using namespace llvm;

  MetaAddress BlockAddress = getMetaAddressMetadata(Call,
                                                    "revng.callerblock.start");
  auto Successors = Function.CFG.at(BlockAddress).Successors;

  // Find the call edge
  model::CallEdge *ModelCall = nullptr;
  for (auto &Edge : Successors) {
    if (auto *CE = dyn_cast<model::CallEdge>(Edge.get())) {
      revng_assert(ModelCall == nullptr);
      ModelCall = CE;
    }
  }
  revng_assert(ModelCall != nullptr);

  model::TypePath PrototypePath = getPrototype(Binary, *ModelCall);
  return dyn_cast_or_null<model::RawFunctionType>(PrototypePath.get());
}
