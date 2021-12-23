#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

namespace model {
inline MetaAddress getMetaAddressOfIsolatedFunction(const llvm::Function &F) {
  revng_assert(FunctionTags::Lifted.isTagOf(&F));
  return getMetaAddressMetadata(&F, FunctionEntryMDNName);
}

inline Function *llvmToModelFunction(Binary &Binary, const llvm::Function &F) {
  auto MaybeMetaAddres = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddres == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions.find(MaybeMetaAddres);
      It != Binary.Functions.end())
    return &(*It);

  return nullptr;
}

inline const Function *
llvmToModelFunction(const Binary &Binary, const llvm::Function &F) {
  auto MaybeMetaAddres = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddres == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions.find(MaybeMetaAddres);
      It != Binary.Functions.end())
    return &*It;

  return nullptr;
}

} // namespace model
