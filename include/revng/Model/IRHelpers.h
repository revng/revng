#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

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

inline MetaAddress getMetaAddressOfIsolatedFunction(const llvm::Function &F) {
  revng_assert(FunctionTags::Lifted.isTagOf(&F));
  return getMetaAddressMetadata(&F, FunctionEntryMDNName);
}

inline model::Function *
llvmToModelFunction(model::Binary &Binary, const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddress == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions.find(MaybeMetaAddress);
      It != Binary.Functions.end())
    return &(*It);

  return nullptr;
}

inline const model::Function *
llvmToModelFunction(const model::Binary &Binary, const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddress == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions.find(MaybeMetaAddress);
      It != Binary.Functions.end())
    return &*It;

  return nullptr;
}
