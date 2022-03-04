//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/Assert.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

template<typename T>
concept DerivedValue = std::is_base_of_v<llvm::Value, T>;

using std::conditional_t;

template<DerivedValue ConstnessT, DerivedValue ResultT>
using PossiblyConstValueT = conditional_t<std::is_const_v<ConstnessT>,
                                          std::add_const_t<ResultT>,
                                          std::remove_const_t<ResultT>>;

template<typename T>
concept PossiblyConstInsertValue = std::is_same_v<std::remove_const_t<T>,
                                                  llvm::InsertValueInst>;

template<DerivedValue T>
using ValueT = PossiblyConstValueT<T, llvm::Value>;

template<PossiblyConstInsertValue T>
llvm::SmallVector<ValueT<T> *, 2>

getConstQualifiedInsertValueLeafOperands(T *Ins) {
  using ValT = ValueT<T>;

  llvm::SmallVector<ValT *, 2> Results;
  llvm::SmallSet<unsigned, 2> FoundIds;

  auto *StructTy = llvm::cast<llvm::StructType>(Ins->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, nullptr);

  // InsertValues should be present in the IR only when we are returning an
  // LLVM struct from a function. In particular, there should be a chain of
  // InsertValues inserting to the same aggregate, followed by a ret of the
  // aggregate value.
  auto IsRet = [](const llvm::Value *V) { return isa<llvm::ReturnInst>(V); };
  auto FirstUserIsInsertVal = [&Ins]() {
    return isa<llvm::InsertValueInst>(Ins->use_begin()->getUser());
  };
  revng_assert((Ins->getNumUses() == 1 and FirstUserIsInsertVal())
               or llvm::all_of(Ins->users(), IsRet));

  while (1) {
    revng_assert(Ins->getNumIndices() == 1);

    // It must be the first time that we insert a value at this index of the
    // aggregate
    unsigned FieldId = Ins->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(FoundIds.count(FieldId) == 0);
    FoundIds.insert(FieldId);

    // Save the inserted value
    ValT *Op = Ins->getInsertedValueOperand();
    revng_assert(isa<llvm::IntegerType>(Op->getType())
                 or isa<llvm::PointerType>(Op->getType()));
    revng_assert(Results[FieldId] == nullptr);
    Results[FieldId] = Op;

    // Go back in the insertValue chain ...
    ValT *Tmp = Ins->getAggregateOperand();
    Ins = llvm::dyn_cast<llvm::InsertValueInst>(Tmp);
    if (not Ins) {
      // ... until you find an undef or constant aggregate (i.e. you have
      // reached the first insertValue of the chain)
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

template<DerivedValue T>
using ExtractValueT = PossiblyConstValueT<T, llvm::ExtractValueInst>;

template<DerivedValue T>
using ExtractValuePtrSet = llvm::SmallPtrSet<ExtractValueT<T> *, 2>;

template<DerivedValue T>
llvm::SmallVector<ExtractValuePtrSet<T>, 2>
getConstQualifiedExtractedValuesFromInstruction(T *I) {

  llvm::SmallVector<ExtractValuePtrSet<T>, 2> Results;

  auto *StructTy = llvm::cast<llvm::StructType>(I->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, {});

  // Find extract value uses transitively, traversing PHIs and markers
  ExtractValuePtrSet<T> ExtractValues;
  for (auto *TheUser : I->users()) {
    if (auto *ExtractV = dyn_cast<llvm::ExtractValueInst>(TheUser)) {
      ExtractValues.insert(ExtractV);
    } else {
      if (auto *Call = dyn_cast<llvm::CallInst>(TheUser)) {
        if (not FunctionTags::Marker.isTagOf(Call->getCalledFunction()))
          continue;
      }

      // traverse PHIS and markers until we find extractvalues
      llvm::SmallPtrSet<ValueT<T> *, 8> Visited = {};
      llvm::SmallPtrSet<ValueT<T> *, 8> ToVisit = { TheUser };
      while (not ToVisit.empty()) {

        llvm::SmallPtrSet<ValueT<T> *, 8> NextToVisit = {};

        for (ValueT<T> *Ident : ToVisit) {
          Visited.insert(Ident);
          NextToVisit.erase(Ident);

          for (auto *User : Ident->users()) {
            if (auto *ExtractV = llvm::dyn_cast<llvm::ExtractValueInst>(User)) {
              ExtractValues.insert(ExtractV);
            } else if (auto *IdentUser = llvm::dyn_cast<llvm::CallInst>(User)) {
              if (FunctionTags::Marker.isTagOf(IdentUser))
                NextToVisit.insert(IdentUser);
            } else if (auto *PHIUser = llvm::dyn_cast<llvm::PHINode>(User)) {
              if (not Visited.count(PHIUser))
                NextToVisit.insert(PHIUser);
            }
          }
        }

        ToVisit = NextToVisit;
      }
    }
  }

  for (auto *E : ExtractValues) {
    revng_assert(E->getNumIndices() == 1);
    unsigned FieldId = E->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    Results[FieldId].insert(E);
  }

  return Results;
};

llvm::SmallVector<llvm::SmallPtrSet<llvm::ExtractValueInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

llvm::SmallVector<llvm::SmallPtrSet<const llvm::ExtractValueInst *, 2>, 2>
getExtractedValuesFromInstruction(const llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}
