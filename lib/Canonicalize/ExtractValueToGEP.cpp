//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

static constexpr const char *ExtractValueToGEPFlag = "extractvalue-to-gep";

struct ExtractValueToGEPPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ExtractValueToGEPPass() : FunctionPass(ID) {}

public:
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

/// Fold an extraction whose aggregate is a constant into the constant field.
///
/// The rest of the pass turns an extraction into a load from the `alloca`
/// holding the aggregate, which needs an instruction producing it. When the
/// aggregate has been folded into a constant there is no such instruction, but
/// the extraction still has to go: `OpaqueExtractValue` is opaque to LLVM, so
/// nothing else ever removes it, and the stages downstream of this pass have no
/// case for it at all.
static bool foldConstantExtractions(llvm::Function &F) {
  SmallVector<Instruction *, 8> Folded;

  for (Instruction &I : llvm::instructions(F)) {
    Constant *Field = nullptr;

    if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
      // Only a single index, as in the rewriting below.
      if (EV->getNumIndices() != 1)
        continue;
      auto *Aggregate = dyn_cast<Constant>(EV->getAggregateOperand());
      if (Aggregate != nullptr)
        Field = Aggregate->getAggregateElement(EV->getIndices()[0]);
    } else if (auto *
                 OpaqueEV = getCallToTagged(&I,
                                            FunctionTags::OpaqueExtractValue)) {
      // An `OpaqueExtractValue` always carries exactly one index.
      auto *Aggregate = dyn_cast<Constant>(OpaqueEV->getArgOperand(0));
      const auto *Index = cast<ConstantInt>(OpaqueEV->getArgOperand(1));
      if (Aggregate != nullptr
          and Index->getValue().ule(std::numeric_limits<unsigned>::max()))
        Field = Aggregate->getAggregateElement(unsigned(Index->getZExtValue()));
    } else {
      continue;
    }

    if (Field == nullptr or Field->getType() != I.getType())
      continue;

    I.replaceAllUsesWith(Field);
    Folded.push_back(&I);
  }

  for (Instruction *I : Folded)
    I->eraseFromParent();

  return not Folded.empty();
}

static auto getStructValuedInstructions(llvm::Function &F) {
  SmallSetVector<Instruction *, 8> StructValues;

  for (Instruction &I : llvm::instructions(F)) {
    Instruction *Struct = nullptr;
    if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
      if (EV->getNumIndices() == 1)
        Struct = dyn_cast<Instruction>(EV->getAggregateOperand());
    } else if (auto *
                 OpaqueEV = getCallToTagged(&I,
                                            FunctionTags::OpaqueExtractValue)) {
      Struct = dyn_cast<Instruction>(OpaqueEV->getArgOperand(0));
    }

    if (Struct)
      StructValues.insert(Struct);
  }

  return StructValues.takeVector();
}

bool ExtractValueToGEPPass::runOnFunction(llvm::Function &F) {
  bool Changed = foldConstantExtractions(F);

  SmallVector<Instruction *> StructValues = getStructValuedInstructions(F);
  if (StructValues.empty())
    return Changed;

  LLVMContext &Context = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  IntegerType *Int8Ty = llvm::IntegerType::getInt8Ty(Context);
  revng::IRBuilder B(Context);

  SmallVector<WeakTrackingVH, 8> IsDead;

  for (Instruction *StructValue : StructValues) {
    auto *TheStructType = cast<StructType>(StructValue->getType());
    const StructLayout *Layout = DL.getStructLayout(TheStructType);

    Value *BasePointer = nullptr;
    if (auto *Load = dyn_cast<LoadInst>(StructValue)) {
      BasePointer = Load->getPointerOperand();
    } else {
      DebugLoc DebugInfo = StructValue->getDebugLoc();
      B.SetInsertPointPastAllocas(&F, DebugInfo);
      BasePointer = B.CreateAlloca(TheStructType);

      B.SetInsertPoint(StructValue->getInsertionPointAfterDef(), DebugInfo);
      B.CreateStore(StructValue, BasePointer);
    }

    for (Use &U : StructValue->uses()) {
      auto *User = U.getUser();

      const llvm::ConstantInt *Index = nullptr;

      if (auto *EV = dyn_cast<ExtractValueInst>(User)) {
        if (EV->getNumIndices() != 1)
          continue;
        unsigned I = *EV->indices().begin();
        Index = ConstantInt::get(IntegerType::getInt64Ty(Context), I);
      } else if (auto *
                   OpaqueEV = getCallToTagged(User,
                                              FunctionTags::OpaqueExtractValue);
                 OpaqueEV) {
        Index = cast<ConstantInt>(OpaqueEV->getArgOperand(1));
      }

      if (not Index)
        continue;

      // The following check could be an assertion, but let's be permissive.
      if (Index->isNegative()
          or Index->uge(std::numeric_limits<unsigned>::max()))
        continue;

      auto *ExtractValue = cast<Instruction>(User);
      B.SetInsertPoint(ExtractValue, ExtractValue->getDebugLoc());

      auto *LoadAddress = BasePointer;

      unsigned FieldId = Index->getZExtValue();
      uint64_t FieldOffset = Layout->getElementOffset(FieldId);
      if (FieldOffset) {
        auto *Offset = llvm::ConstantInt::get(Index->getType(),
                                              APInt(64, FieldOffset));
        LoadAddress = B.CreateGEP(Int8Ty, BasePointer, Offset);
      }
      auto *Load = B.CreateLoad(ExtractValue->getType(), LoadAddress);
      ExtractValue->replaceAllUsesWith(Load);
      IsDead.push_back(ExtractValue);
    }
  }

  // The following call to RecursivelyDeleteTriviallyDeadInstructionsPermissive
  // cannot remove dead OpaqueExtractValue, so we have to do that ourselves.
  // In principle we could change the attributes of the OpaqueExtractValue
  // functions but that would enable DCE to remove them in other places where we
  // need them to survive.
  for (WeakTrackingVH &VH : IsDead) {
    if (VH.pointsToAliveValue()) {
      if (auto *OpaqueEV = getCallToTagged(&*VH,
                                           FunctionTags::OpaqueExtractValue)) {
        revng_assert(OpaqueEV);
        revng_assert(0 == OpaqueEV->getNumUses());
        OpaqueEV->eraseFromParent();
      }
    }
  }
  // After manually removing the OpaqueExtractValue, let LLVM do its job.
  RecursivelyDeleteTriviallyDeadInstructions(IsDead);

  return true;
}

char ExtractValueToGEPPass::ID = 0;

static constexpr const char *Description = "extractvalue-to-i8-GEP replacement";

static llvm::RegisterPass<ExtractValueToGEPPass> X{ ExtractValueToGEPFlag,
                                                    Description,
                                                    false,
                                                    false };
