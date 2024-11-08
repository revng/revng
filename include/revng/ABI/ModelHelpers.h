#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Type.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
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
inline llvm::IntegerType *getPointerSizedInteger(llvm::LLVMContext &C,
                                                 const model::Binary &Binary) {
  const size_t PtrSize = getPointerSize(Binary.Architecture());
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

extern llvm::SmallVector<model::UpcastableType>
flattenReturnTypes(const abi::FunctionType::Layout &Layout,
                   const model::Binary &Model);
