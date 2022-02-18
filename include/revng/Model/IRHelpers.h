#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

/// \brief Given a Call instruction and the model type of its parent function,
///        return the edge on the model that represents that call, or nullptr if
///        this doesn't exist.
inline model::CallEdge *getCallEdge(const model::Binary &Binary,
                                    const model::Function &Function,
                                    const llvm::CallInst *Call) {
  using namespace llvm;

  MetaAddress BlockAddress = getMetaAddressMetadata(Call,
                                                    CallerBlockStartMDName);
  if (BlockAddress.isInvalid())
    return nullptr;

  auto &Successors = Function.CFG.at(BlockAddress).Successors;

  // Find the call edge
  model::CallEdge *ModelCall = nullptr;
  for (auto &Edge : Successors) {
    if (auto *CE = dyn_cast<model::CallEdge>(Edge.get())) {
      revng_assert(ModelCall == nullptr);
      ModelCall = CE;
    }
  }
  revng_assert(ModelCall != nullptr);

  return ModelCall;
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

/// \return the prototype associated to a CallInst.
///
/// \note If the model type of the parent function is not provided, this will be
///       deduced using the Call instruction's parent function.
///
/// \note If the callsite has no associated prototype, e.g. the called functions
///       is not an isolated function, a null pointer is returned.
inline model::Type *
getCallSitePrototype(const model::Binary &Binary,
                     const llvm::CallInst *Call,
                     const model::Function *ParentFunction = nullptr) {
  if (not ParentFunction)
    ParentFunction = llvmToModelFunction(Binary, *Call->getFunction());

  if (not ParentFunction)
    return nullptr;

  const model::CallEdge *ModelCallEdge = getCallEdge(Binary,
                                                     *ParentFunction,
                                                     Call);
  if (not ModelCallEdge)
    return nullptr;

  return getPrototype(Binary, *ModelCallEdge).get();
}
