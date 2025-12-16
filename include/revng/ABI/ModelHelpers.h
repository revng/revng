#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/TypeDefinition.h"

namespace llvm {
class CallInst;
class Use;
} // namespace llvm

/// Get the model type of an llvm::Value
///
/// \p V must be of IntegerType or an AllocaInst/GlobalVariable of IntegerType
/// or ArrayType.
///
/// \return a valid model type
extern model::UpcastableType modelType(const llvm::Value *V,
                                       const model::Binary &Model);

/// Convert an LLVM integer type (i1, i8, i16, ...) to the corresponding
/// primitive type (uint8_t, uint16_t, ...).
extern model::UpcastableType llvmIntToModelType(const llvm::Type *LLVMType,
                                                const model::Binary &Model);

/// Try to extract a model type from an llvm::Value. V must be a pointer to
/// a string which contains a valid serialization of the type, otherwise
/// this function will abort.
extern model::UpcastableType fromLLVMString(llvm::Value *V,
                                            const model::Binary &Model);

/// Create a global string in the given LLVM module that contains a
/// serialization of \a Type.
llvm::Constant *toLLVMString(const model::UpcastableType &Type,
                             llvm::Module &M);

/// Return an LLVM IntegerType that has the size of a pointer in the given
/// architecture.
inline llvm::IntegerType *
getPointerSizedInteger(llvm::LLVMContext &C,
                       const model::Architecture::Values &Architecture) {
  const size_t PtrSize = getPointerSize(Architecture);
  return llvm::Type::getIntNTy(C, PtrSize * 8);
}

/// If possible, deduce the model type returned by \a Inst by looking only at
/// the instruction (e.g. ModelGEPs). Note that calls to RawFunctionTypes and
/// calls to StructInitializer can return more than one type.
/// \return nothing if no information could be deduced locally on Inst
/// \return one or more types associated to the instruction
extern RecursiveCoroutine<llvm::SmallVector<model::UpcastableType, 8>>
getStrongModelInfo(const llvm::Instruction *Inst, const model::Binary &Model);

/// If possible, deduce the expected model type of an operand (e.g. the base
/// operand of a ModelGEP) by looking only at the User. Note that, in the case
/// of `ret` instructions inside RawFunctionTypes, the use might have more than
/// one type associated to it.
/// \return nothing if no information could be deduced locally on U
/// \return one or more types associated to the use
extern llvm::SmallVector<model::UpcastableType>
getExpectedModelType(const llvm::Use *U, const model::Binary &Model);

/// \note This is the final prototype, after segregate-stack-access
template<bool LegacyLocalVariables>
inline llvm::FunctionType &
layoutToLLVMFunctionType(llvm::LLVMContext &Context,
                         model::Architecture::Values Architecture,
                         const abi::FunctionType::Layout &Layout) {
  using namespace llvm;

  // Process arguments
  using namespace abi::FunctionType;
  SmallVector<Type *> FunctionArguments;
  for (const Layout::Argument &Argument : Layout.Arguments) {
    model::UpcastableType ArgumentType = Argument.Type;

    switch (Argument.Kind) {
    case abi::FunctionType::ArgumentKind::ShadowPointerToAggregateReturnValue:
      // Skip SPTAR
      continue;
    case abi::FunctionType::ArgumentKind::PointerToCopy:
    case abi::FunctionType::ArgumentKind::ReferenceToAggregate:
      ArgumentType = model::PointerType::make(std::move(ArgumentType),
                                              Architecture);
      break;
    case abi::FunctionType::ArgumentKind::Scalar:
      // Do nothing
      break;
    default:
      revng_abort();
    }

    auto *LLVMType = getLLVMTypeForScalar(Context, *ArgumentType);
    FunctionArguments.push_back(LLVMType);
  }

  // Process return type
  Type *ReturnType = nullptr;
  switch (Layout.returnMethod()) {
  case ReturnMethod::Void:
    // No return values, use void
    ReturnType = Type::getVoidTy(Context);
    break;

  case ReturnMethod::ModelAggregate: {
    auto TargetPointerSizedInteger = getPointerSizedInteger(Context,
                                                            Architecture);
    if constexpr (LegacyLocalVariables) {
      ReturnType = TargetPointerSizedInteger;
    } else {
      const model::Type &ReturnAggregate = Layout.returnValueAggregateType();
      size_t ReturnSize = *ReturnAggregate.size();
      auto *Int8 = llvm::IntegerType::getInt8Ty(Context);
      ReturnType = llvm::ArrayType::get(Int8, ReturnSize);
    }
  } break;

  case ReturnMethod::Scalar: {
    // We either have a return value that fits in a single register, or it's
    // CABIFunctionDefinition returning stuff through registers
    unsigned Bits = 0;
    for (const Layout::ReturnValue &ReturnValue : Layout.ReturnValues)
      Bits += ReturnValue.Type->size().value() * 8;
    ReturnType = IntegerType::getIntNTy(Context, Bits);
  } break;

  case ReturnMethod::RegisterSet: {
    // We have a RawFunctionDefinition returning things over multiple
    // registers
    revng_assert(Layout.returnValueRegisterCount() > 1);
    auto Types = toLLVMTypes(Context, Layout.returnValueRegisters());
    ReturnType = StructType::get(Context, Types, true);
  } break;

  default:
    revng_abort();
  }

  return *FunctionType::get(ReturnType, FunctionArguments, false);
}
