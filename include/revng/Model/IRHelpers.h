#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

inline MetaAddress getMetaAddressOfIsolatedFunction(const llvm::Function &F) {
  revng_assert(FunctionTags::Isolated.isTagOf(&F));
  return getMetaAddressMetadata(&F, FunctionEntryMDNName);
}

inline model::Function *llvmToModelFunction(model::Binary &Binary,
                                            const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddress == MetaAddress::invalid())
    return nullptr;
  if (auto It = Binary.Functions().find(MaybeMetaAddress);
      It != Binary.Functions().end())
    return &*It;

  return nullptr;
}

inline const model::Function *llvmToModelFunction(const model::Binary &Binary,
                                                  const llvm::Function &F) {
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDNName);
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

/// Create an empty model::StructType of size Size in Binary
inline model::TypePath createEmptyStruct(model::Binary &Binary, uint64_t Size) {
  using namespace model;

  revng_assert(Size > 0 and Size < std::numeric_limits<int64_t>::max());
  TypePath Path = Binary.makeType<model::StructType>().second;
  model::StructType *NewStruct = llvm::cast<model::StructType>(Path.get());
  NewStruct->Size() = Size;
  return Path;
}
