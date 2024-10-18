#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

constexpr const char *PrototypeMDName = "revng.prototype";

template<ConstOrNot<model::Binary> T>
inline ConstPtrIfConst<T, model::TypeDefinition>
getCallSitePrototype(T &Binary, const llvm::Instruction *Call) {
  revng_assert(llvm::isa<llvm::CallInst>(Call));

  llvm::StringRef SerializedRef = fromStringMetadata(Call, PrototypeMDName);
  auto Result = model::DefinitionReference::fromString(&Binary, SerializedRef);

  if constexpr (std::is_const_v<T>)
    return Result.getConst();
  else
    return Result.get();
}

inline model::Function *llvmToModelFunction(model::Binary &Binary,
                                            const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDName);
  if (MaybeMetaAddress == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions().tryGet(MaybeMetaAddress); It != nullptr)
    return It;

  return nullptr;
}

inline const model::Function *llvmToModelFunction(const model::Binary &Binary,
                                                  const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDName);
  if (MaybeMetaAddress == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions().find(MaybeMetaAddress);
      It != Binary.Functions().end())
    return &*It;

  return nullptr;
}

inline llvm::IntegerType *getLLVMIntegerTypeFor(llvm::LLVMContext &Context,
                                                const model::Type &Type) {
  revng_assert(Type.size());
  return llvm::IntegerType::getIntNTy(Context, *Type.size() * 8);
}

inline llvm::IntegerType *getLLVMTypeForScalar(llvm::LLVMContext &Context,
                                               const model::Type &Type) {
  revng_assert(Type.isScalar());
  return getLLVMIntegerTypeFor(Context, Type);
}

inline std::string getLLVMFunctionName(const model::Function &Function) {
  std::string Result = "local_";
  if (DebugNames)
    Result += Function.name().str().str();
  else
    Result += Function.Entry().toString();

  return Result;
}
