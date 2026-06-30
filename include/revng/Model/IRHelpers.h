#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/Tag.h"

constexpr const char *PrototypeMDName = "revng.prototype";

[[nodiscard]] inline std::string llvmName(const model::Function &Function) {
  if (DebugNames and not Function.Name().empty())
    return "local_" + Function.Name();
  else
    return "local_" + Function.Entry().toIdentifier();
}

template<ConstOrNot<model::Binary> T>
inline ConstPtrIfConst<T, model::TypeDefinition>
getCallSitePrototype(T &Binary, const llvm::Instruction *Call) {
  revng_assert(llvm::isa<llvm::CallInst>(Call));

  llvm::StringRef SerializedRef = fromStringMetadata(Call, PrototypeMDName);
  auto Result = model::TypeDefinitionReference::fromString(&Binary,
                                                           SerializedRef);

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

/// Given an LLVM module with a single isolated function, find it and return a
/// reference to it. This function will assert if there is any other amount (0,
/// 2 -> inf) number of functions. If the Address is provided it will also
/// assert that the found function has the expected MetaAddress.
inline decltype(auto)
getUniqueIsolatedFunction(ConstOrNot<llvm::Module> auto &Module,
                          const MetaAddress &Address = MetaAddress::invalid()) {
  using ModuleType = std::remove_reference_t<decltype(Module)>;
  using FunctionType = ConstIf<std::is_const_v<ModuleType>, llvm::Function>;

  FunctionType *Function = nullptr;
  for (FunctionType &F : Module.functions()) {
    if (F.isDeclaration() or not FunctionTags::Isolated.isTagOf(&F))
      continue;

    revng_assert(Function == nullptr);
    Function = &F;
  }
  revng_assert(Function != nullptr);

  if (not Address.isInvalid()) {
    MetaAddress FoundAddress = getMetaAddressOfIsolatedFunction(*Function);
    revng_assert(Address == FoundAddress);
  }

  return *Function;
}

inline llvm::SmallVector<llvm::Type *>
toLLVMTypes(llvm::LLVMContext &Context,
            const llvm::SmallVector<model::Register::Values> &Registers) {
  using namespace llvm;
  SmallVector<llvm::Type *> Result;
  auto IntoLLVMType = [&Context](model::Register::Values V) -> Type * {
    return IntegerType::getIntNTy(Context, 8 * model::Register::getSize(V));
  };
  std::ranges::copy(Registers | std::views::transform(IntoLLVMType),
                    std::back_inserter(Result));
  return Result;
}

namespace SegmentGlobal {

constexpr const char *StartAddressMDName = "revng.segment.start";

inline std::string getNameFor(const MetaAddress &StartAddress) {
  return "segment_" + StartAddress.toIdentifier();
}

inline std::string getNameFor(const model::Segment &Segment) {
  return getNameFor(Segment.StartAddress());
}

inline MetaAddress getAddress(const llvm::GlobalVariable &Variable) {
  return getMetaAddressMetadata(&Variable, StartAddressMDName);
}

inline llvm::GlobalVariable &
get(llvm::Module &M, const MetaAddress &StartAddress, uint64_t VirtualSize) {
  using namespace llvm;

  auto *T = llvm::ArrayType::get(IntegerType::getInt8Ty(M.getContext()),
                                 VirtualSize);

  // TODO: should global variables for read-only segments be constant?
  auto &Result = getOrCreateGlobal(M,
                                   getNameFor(StartAddress),
                                   T,
                                   false,
                                   GlobalValue::ExternalLinkage);

  FunctionTags::SegmentGlobal.addTo(&Result);
  setMetaAddressMetadata(&Result, StartAddressMDName, StartAddress);

  return Result;
}

inline llvm::GlobalVariable &get(llvm::Module &M,
                                 const model::Segment &Segment) {
  return get(M, Segment.StartAddress(), Segment.VirtualSize());
}

} // namespace SegmentGlobal
