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

inline MetaAddress getMetaAddressOfIsolatedFunction(const llvm::Function &F) {
  revng_assert(FunctionTags::Isolated.isTagOf(&F));
  return getMetaAddressMetadata(&F, FunctionEntryMDName);
}

inline model::TypeDefinitionPath
getCallSitePrototype(ConstOrNot<model::Binary> auto &Root,
                     const llvm::Instruction *Call) {
  using namespace llvm;
  revng_assert(isa<CallInst>(Call));

  using TDP = model::TypeDefinitionPath;
  return TDP::fromString(&Root, fromStringMetadata(Call, PrototypeMDName));
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

inline llvm::IntegerType *
getLLVMIntegerTypeFor(llvm::LLVMContext &Context,
                      const model::QualifiedType &QT) {
  revng_assert(QT.size());
  return llvm::IntegerType::getIntNTy(Context, *QT.size() * 8);
}

inline llvm::IntegerType *getLLVMTypeForScalar(llvm::LLVMContext &Context,
                                               const model::QualifiedType &QT) {
  revng_assert(QT.isScalar());
  return getLLVMIntegerTypeFor(Context, QT);
}

/// Create an empty model::StructDefinition of size Size in Binary
inline model::TypeDefinitionPath createEmptyStruct(model::Binary &Binary,
                                                   uint64_t Size) {
  revng_assert(Size > 0 and Size < std::numeric_limits<int64_t>::max());

  auto [Struct, Path] = Binary.makeTypeDefinition<model::StructDefinition>();
  Struct.Size() = Size;
  return Path;
}

inline std::string getLLVMFunctionName(const model::Function &Function) {
  std::string Result = "local_";
  if (DebugNames)
    Result += Function.name().str().str();
  else
    Result += Function.Entry().toString();

  return Result;
}
