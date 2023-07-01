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

template<DerivedValue T>
using ValueT = PossiblyConstValueT<T, llvm::Value>;

template<DerivedValue T>
using CallT = PossiblyConstValueT<T, llvm::CallInst>;

template<DerivedValue T>
using CallPtrSet = llvm::SmallPtrSet<CallT<T> *, 2>;

template<DerivedValue T>
llvm::SmallVector<CallPtrSet<T>, 2>
getConstQualifiedExtractedValuesFromInstruction(T *I) {

  llvm::SmallVector<CallPtrSet<T>, 2> Results;

  auto *StructTy = llvm::cast<llvm::StructType>(I->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, {});

  // Find extract value uses transitively, traversing PHIs and markers
  CallPtrSet<T> Calls;
  for (auto *TheUser : I->users()) {
    if (auto *ExtractV = getCallToTagged(TheUser,
                                         FunctionTags::OpaqueExtractValue)) {
      Calls.insert(ExtractV);
    } else {
      if (auto *Call = dyn_cast<llvm::CallInst>(TheUser)) {
        if (not isCallToTagged(Call, FunctionTags::Parentheses))
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
            using FunctionTags::OpaqueExtractValue;
            if (auto *EV = getCallToTagged(User, OpaqueExtractValue)) {
              Calls.insert(EV);
            } else if (auto *IdentUser = llvm::dyn_cast<llvm::CallInst>(User)) {
              if (isCallToTagged(IdentUser, FunctionTags::Parentheses))
                NextToVisit.insert(IdentUser);
            } else if (auto *PHIUser = llvm::dyn_cast<llvm::PHINode>(User)) {
              if (not Visited.contains(PHIUser))
                NextToVisit.insert(PHIUser);
            }
          }
        }

        ToVisit = NextToVisit;
      }
    }
  }

  for (auto *E : Calls) {
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    auto FieldId = cast<llvm::ConstantInt>(E->getArgOperand(1))->getZExtValue();
    Results[FieldId].insert(E);
  }

  return Results;
};

llvm::SmallVector<llvm::SmallPtrSet<llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

llvm::SmallVector<llvm::SmallPtrSet<const llvm::CallInst *, 2>, 2>
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

  auto &Context = SegmentRefFunction.getContext();

  QuickMetadata QMD(Context);

  auto *SAMD = QMD.get(StartAddress.toString());
  revng_assert(SAMD != nullptr);

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Context), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  SegmentRefFunction.setMetadata(SegmentRefMDName, QMD.tuple({ SAMD, VSMD }));
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

  auto *SAMD = cast<MDString>(Node->getOperand(0));
  MetaAddress StartAddress = MetaAddress::fromString(SAMD->getString());
  auto *VSMD = cast<ConstantAsMetadata>(Node->getOperand(1))->getValue();
  uint64_t VirtualSize = cast<ConstantInt>(VSMD)->getZExtValue();

  return { StartAddress, VirtualSize };
}

void setStringLiteralMetadata(llvm::Function &StringLiteralFunction,
                              MetaAddress StartAddress,
                              uint64_t VirtualSize,
                              uint64_t Offset,
                              uint64_t StringLength) {
  using namespace llvm;

  auto *M = StringLiteralFunction.getParent();
  auto &Ctx = StringLiteralFunction.getContext();

  QuickMetadata QMD(M->getContext());
  auto StringLiteralMDKind = Ctx.getMDKindID(StringLiteralMDName);

  Constant *SAConstant = StartAddress.toValue(M);
  auto *SAMD = ConstantAsMetadata::get(SAConstant);

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Ctx), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  auto *OffsetConstant = ConstantInt::get(Type::getInt64Ty(Ctx), Offset);
  auto *OffsetMD = ConstantAsMetadata::get(OffsetConstant);

  auto *StrLenConstant = ConstantInt::get(Type::getInt64Ty(Ctx), StringLength);
  auto *StrLenMD = ConstantAsMetadata::get(StrLenConstant);

  auto QMDTuple = QMD.tuple({ SAMD, VSMD, OffsetMD, StrLenMD });
  StringLiteralFunction.setMetadata(StringLiteralMDKind, QMDTuple);
}

bool hasStringLiteralMetadata(const llvm::Function &F) {
  auto &Ctx = F.getContext();
  auto StringLiteralMDKind = Ctx.getMDKindID(StringLiteralMDName);
  return nullptr != F.getMetadata(StringLiteralMDKind);
}

std::tuple<MetaAddress, uint64_t, uint64_t, uint64_t>
extractStringLiteralFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasStringLiteralMetadata(F));

  auto &Ctx = F.getContext();

  auto StringLiteralMDKind = Ctx.getMDKindID(StringLiteralMDName);
  auto *Node = F.getMetadata(StringLiteralMDKind);

  auto *SAMD = cast<ConstantAsMetadata>(Node->getOperand(0))->getValue();
  auto *SAConstant = cast<Constant>(SAMD);
  MetaAddress StartAddress = MetaAddress::fromValue(SAConstant);
  auto *VSMD = cast<ConstantAsMetadata>(Node->getOperand(1))->getValue();
  uint64_t VirtualSize = cast<ConstantInt>(VSMD)->getZExtValue();
  auto *OffsetMD = cast<ConstantAsMetadata>(Node->getOperand(2))->getValue();
  uint64_t Offset = cast<ConstantInt>(OffsetMD)->getZExtValue();
  auto *StrLenMD = cast<ConstantAsMetadata>(Node->getOperand(3))->getValue();
  uint64_t StrLen = cast<ConstantInt>(StrLenMD)->getZExtValue();

  return { StartAddress, VirtualSize, Offset, StrLen };
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

  Builder.CreateCall(Callee, getUniqueString(M, "emitMessage", Message.str()));
}
