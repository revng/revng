#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Type.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

namespace llvm {
class CallInst;
class Use;
} // namespace llvm

/// Strip off all the possible layers of constness and typedefs from QT
extern model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT);

/// Get the model type of an llvm::Value
///
/// \p V must be of IntegerType or an AllocaInst/GlobalVariable of IntegerType
/// or ArrayType.
///
/// \return a valid QualifiedType
extern const model::QualifiedType
modelType(const llvm::Value *V, const model::Binary &Model);

/// Convert an LLVM integer type (i1, i8, i16, ...) to the corresponding
/// primitive type (uint8_t, uint16_t, ...).
extern const model::QualifiedType
llvmIntToModelType(const llvm::Type *LLVMType, const model::Binary &Model);

/// Tries to extract a QualifiedType from an llvm::Value. V must be a pointer to
/// a string which contains a valid serialization of a QualifiedType, otherwise
/// this function will abort.
extern model::QualifiedType
deserializeFromLLVMString(llvm::Value *V, const model::Binary &Model);

/// Create a global string in the given LLVM module that contains a
/// serialization of \a QT.
llvm::Constant *
serializeToLLVMString(const model::QualifiedType &QT, llvm::Module &M);

/// Return an LLVM IntegerType that has the size of a pointer in the given
/// architecture.
inline llvm::IntegerType *
getPointerSizedInteger(llvm::LLVMContext &C, const model::Binary &Binary) {
  const size_t PtrSize = getPointerSize(Binary.Architecture());
  return llvm::Type::getIntNTy(C, PtrSize * 8);
}

/// Create a pointer to the given base Type
inline model::QualifiedType
createPointerTo(const model::TypePath &BaseT, const model::Binary &Binary) {
  using Qualifier = model::Qualifier;

  return model::QualifiedType{
    BaseT, { Qualifier::createPointer(Binary.Architecture()) }
  };
}

/// Drops te last pointer qualifier from \a QT or, if this wraps a
/// typedef, it recursively descends the typedef wrappers until a pointer
/// qualifier is found.
extern RecursiveCoroutine<model::QualifiedType>
dropPointer(const model::QualifiedType &QT);

/// Returns the type of the Idx-th field of a QualifiedType. This aborts if
/// the type has no fields (e.g. pointers or primitive types) or if Idx is
// out-of-bounds. Typedefs are traversed transparently.
extern RecursiveCoroutine<model::QualifiedType>
getFieldType(const model::QualifiedType &Parent, uint64_t Idx);

/// Returns the type of the Idx-th field of a QualifiedType. If \a Idx is a
/// constant int, it behaves like `getFieldType(Parent, uint64_t)`.
/// If \a Idx is any other llvm::Value, \a Parent must be an array.
extern model::QualifiedType
getFieldType(const model::QualifiedType &Parent, llvm::Value *Idx);

/// Starting from \a Base, traverse the type-system at an arbitrary depth using
/// \a Indexes.
extern model::QualifiedType
traverseTypeSystem(const model::QualifiedType &Base,
                   const llvm::SmallVector<llvm::Value *, 8> &Indexes);

/// Calculate the model type that is returned by a ModelGEP.
/// \param Model The binary on which we are operating
/// \param Call The ModelGEP (or ModelGEP-like) Call
/// \param BaseType The starting type for the type-system traversal. If none is
///        provided, it will be inferred by the first argument of \a Call.
extern model::QualifiedType
traverseModelGEP(const model::Binary &Model, const llvm::CallInst *Call);

/// If possible, deduce the model type returned by \a Inst by looking only at
/// the instruction (e.g. ModelGEPs). Note that calls to RawFunctionTypes and
/// calls to StructInitializer can return more than one type.
/// \return nothing if no information could be deduced locally on Inst
/// \return one or more QualifiedTypes associated to Inst
extern RecursiveCoroutine<llvm::SmallVector<model::QualifiedType>>
getStrongModelInfo(FunctionMetadataCache &Cache,
                   const llvm::Instruction *Inst,
                   const model::Binary &Model);

/// If possible, deduce the expected model type of an operand (e.g. the base
/// operand of a ModelGEP) by looking only at the User. Note that, in the case
/// of `ret` instructions inside RawFunctionTypes, the use might have more than
/// one type associated to it.
/// \return nothing if no information could be deduced locally on U
/// \return one or more QualifiedTypes associated to U
extern llvm::SmallVector<model::QualifiedType>
getExpectedModelType(FunctionMetadataCache &Cache,
                     const llvm::Use *U,
                     const model::Binary &Model);

extern llvm::SmallVector<model::QualifiedType>
flattenReturnTypes(const abi::FunctionType::Layout &Layout,
                   const model::Binary &Model);

inline model::QualifiedType stripPointer(const model::QualifiedType &Type) {
  model::QualifiedType Result = Type;
  revng_assert(not Result.Qualifiers().empty()
               and model::Qualifier::isPointer(Result.Qualifiers().front()));
  Result.Qualifiers().erase(Result.Qualifiers().begin());
  return Result;
}
