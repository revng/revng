//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/Assert.h"
#include "revng/Support/MetaAddress.h"

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

bool deleteOnlyBody(llvm::Function &F) {
  bool Result = false;
  if (not F.empty()) {
    // deleteBody() also kills all attributes and tags. Since we still
    // want them, we have to save them and re-add them after deleting the
    // body of the function.
    auto Attributes = F.getAttributes();
    auto FTags = FunctionTags::TagsSet::from(&F);

    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> AllMetadata;
    if (F.hasMetadata())
      F.getAllMetadata(AllMetadata);

    // Kill the body.
    F.deleteBody();

    // Restore tags and attributes
    FTags.set(&F);
    F.setAttributes(Attributes);

    F.clearMetadata();
    for (const auto &[KindID, MetaData] : AllMetadata) {
      // Debug metadata is not stripped away by deleteBody() nor by
      // clearMetadata(), but it is wrong to set it twice (the Module would not
      // verify anymore). Hence set the metadata only if its not a debug
      // metadata.
      if (not F.hasMetadata(KindID) and KindID != llvm::LLVMContext::MD_dbg)
        F.setMetadata(KindID, MetaData);
    }

    Result = true;
  }
  return Result;
}

void setSegmentKeyMetadata(llvm::Function &SegmentRefFunction,
                           MetaAddress StartAddress,
                           uint64_t VirtualSize) {
  using namespace llvm;

  auto *M = SegmentRefFunction.getParent();
  auto &Ctx = SegmentRefFunction.getContext();

  QuickMetadata QMD(M->getContext());
  auto SegmentRefMDKind = Ctx.getMDKindID(SegmentRefMDName);

  StructType *MetaAddressTy = MetaAddress::getStruct(M);
  Constant *SAConstant = StartAddress.toConstant(MetaAddressTy);
  auto *SAMD = ConstantAsMetadata::get(SAConstant);

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Ctx), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  SegmentRefFunction.setMetadata(SegmentRefMDKind, QMD.tuple({ SAMD, VSMD }));
}

bool hasSegmentKeyMetadata(const llvm::Function &F) {
  auto &Ctx = F.getContext();
  auto SegmentRefMDKind = Ctx.getMDKindID(SegmentRefMDName);
  return nullptr != F.getMetadata(SegmentRefMDKind);
}

std::pair<MetaAddress, uint64_t>
extractSegmentKeyFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasSegmentKeyMetadata(F));

  auto &Ctx = F.getContext();

  auto SegmentRefMDKind = Ctx.getMDKindID(SegmentRefMDName);
  auto *Node = F.getMetadata(SegmentRefMDKind);

  auto *SAMD = cast<ConstantAsMetadata>(Node->getOperand(0))->getValue();
  auto *SAConstant = cast<Constant>(SAMD);
  MetaAddress StartAddress = MetaAddress::fromConstant(SAConstant);
  auto *VSMD = cast<ConstantAsMetadata>(Node->getOperand(1))->getValue();
  uint64_t VirtualSize = cast<ConstantInt>(VSMD)->getZExtValue();

  return { StartAddress, VirtualSize };
}

void emitMessage(llvm::Instruction *EmitBefore, const llvm::Twine &Message) {
  llvm::IRBuilder<> Builder(EmitBefore);
  emitMessage(Builder, Message);
}

void emitMessage(llvm::IRBuilder<> &Builder, const llvm::Twine &Message) {
  using namespace llvm;

  Module *M = getModule(Builder.GetInsertBlock());
  auto *FT = createFunctionType<void, const uint8_t *>(M->getContext());
  // TODO: use reserved prefix
  llvm::StringRef MessageFunctionName("revng_message");
  FunctionCallee Callee = M->getOrInsertFunction(MessageFunctionName, FT);

  Function *F = cast<Function>(Callee.getCallee());
  if (not FunctionTags::Helper.isTagOf(F))
    FunctionTags::Helper.addTo(F);

  Builder.CreateCall(Callee,
                     getUniqueString(M, "emitMessage", Message.str(), ""));
}
