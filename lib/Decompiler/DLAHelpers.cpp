//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAHelpers.h"

#include "DLATypeSystem.h"

using std::conditional_t;

template<typename ValT>
using LLVMValueT = conditional_t<std::is_const_v<ValT>,
                                 const llvm::Value,
                                 llvm::Value>;

template<typename T>
std::enable_if_t<std::is_same_v<std::remove_const_t<T>, llvm::InsertValueInst>,
                 llvm::SmallVector<LLVMValueT<T> *, 2>>
getConstQualifiedInsertValueLeafOperands(T *Ins) {
  using ValueT = LLVMValueT<T>;
  llvm::SmallVector<ValueT *, 2> Results;
  llvm::SmallSet<unsigned, 2> FoundIds;
  auto *StructTy = llvm::cast<llvm::StructType>(Ins->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, nullptr);
  revng_assert(Ins->getNumUses() == 1
               and (isa<llvm::InsertValueInst>(Ins->use_begin()->getUser())
                    or isa<llvm::ReturnInst>(Ins->use_begin()->getUser())));
  while (1) {
    revng_assert(Ins->getNumIndices() == 1);
    unsigned FieldId = Ins->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(FoundIds.count(FieldId) == 0);
    FoundIds.insert(FieldId);
    ValueT *Op = Ins->getInsertedValueOperand();
    revng_assert(isa<llvm::IntegerType>(Op->getType())
                 or isa<llvm::PointerType>(Op->getType()));
    revng_assert(Results[FieldId] == nullptr);
    Results[FieldId] = Op;
    ValueT *Tmp = Ins->getAggregateOperand();
    Ins = llvm::dyn_cast<llvm::InsertValueInst>(Tmp);
    if (not Ins) {
      revng_assert(llvm::isa<llvm::UndefValue>(Tmp)
                   or llvm::isa<llvm::ConstantAggregate>(Tmp));
      break;
    }
  }
  return Results;
};

llvm::SmallVector<llvm::Value *, 2>
getInsertValueLeafOperands(llvm::InsertValueInst *Ins) {
  return getConstQualifiedInsertValueLeafOperands(Ins);
}

llvm::SmallVector<const llvm::Value *, 2>
getInsertValueLeafOperands(const llvm::InsertValueInst *Ins) {
  return getConstQualifiedInsertValueLeafOperands(Ins);
}

template<typename T>
std::enable_if_t<std::is_same_v<std::remove_const_t<T>, llvm::CallInst>,
                 llvm::SmallVector<LLVMValueT<T> *, 2>>
getConstQualifiedExtractedValuesFromCall(T *Call) {
  using ValueT = LLVMValueT<T>;
  llvm::SmallVector<ValueT *, 2> Results;
  llvm::SmallSet<unsigned, 2> FoundIds;
  auto *StructTy = llvm::cast<llvm::StructType>(Call->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, nullptr);
  revng_assert(Call->getNumUses() <= NumFields);
  for (auto *Extract : Call->users()) {
    auto *E = cast<llvm::ExtractValueInst>(Extract);
    revng_assert(E->getNumIndices() == 1);
    unsigned FieldId = E->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(FoundIds.count(FieldId) == 0);
    FoundIds.insert(FieldId);
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    revng_assert(Results[FieldId] == nullptr);
    Results[FieldId] = E;
  }
  return Results;
};

llvm::SmallVector<llvm::Value *, 2>
getExtractedValuesFromCall(llvm::CallInst *Call) {
  return getConstQualifiedExtractedValuesFromCall(Call);
}

llvm::SmallVector<const llvm::Value *, 2>
getExtractedValuesFromCall(const llvm::CallInst *Call) {
  return getConstQualifiedExtractedValuesFromCall(Call);
}

uint64_t getLoadStoreSizeFromPtrOpUse(const dla::LayoutTypeSystem &TS,
                                      const llvm::Use *U) {
  llvm::Value *AddrOperand = U->get();
  auto *PtrTy = cast<llvm::PointerType>(AddrOperand->getType());
  llvm::Type *AccessedT = PtrTy->getElementType();
  const llvm::DataLayout &DL = TS.getModule().getDataLayout();
  return DL.getTypeAllocSize(AccessedT);
};
