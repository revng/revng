//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <utility>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/PromoteStackPointer/Markers.h"
#include "revng/PromoteStackPointer/PromoteStackPointer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"

// This name is not present after `promote-stack-pointer`.

using namespace llvm;

static Logger Log("promote-stack-pointer");

static void adjustStackAfterCalls(const model::Binary &Binary,
                                  Function &F,
                                  GlobalVariable *GlobalSP) {
  revng::IRBuilder B(F.getParent()->getContext());

  Type *SPType = GlobalSP->getValueType();

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isCallToIsolatedFunction(&I)) {
        auto &ProtoT = *getCallSitePrototype(Binary, cast<CallInst>(&I));
        uint64_t FinalStackOffset = abi::FunctionType::finalStackOffset(ProtoT);
        auto *FSO = ConstantInt::get(SPType, FinalStackOffset);

        // We found a function call
        B.SetInsertPoint(I.getNextNode());
        B.CreateStore(B.CreateAdd(B.createLoad(GlobalSP), FSO), GlobalSP);
      }
    }
  }

  return;
}

namespace revng::pypeline::piperuns {

void PromoteStackPointer::runOnLLVMFunction(const model::Function &Function,
                                            llvm::Function &LLVMFunction) {
  GeneratedCodeBasicInfo GCBI(Binary, *LLVMFunction.getParent());

  {
    // A couple of preliminary assertions
    using namespace FunctionTags;
    revng_assert(TagsSet::from(&LLVMFunction).contains(Isolated));
    revng_assert(not LLVMFunction.isDeclaration());
  }

  // Get the global variable representing the stack pointer register.
  GlobalVariable *GlobalSP = GCBI.spReg();

  if (not GlobalSP) {
    revng_log(Log, "WARNING: cannot find global variable for stack pointer");
    return;
  }

  adjustStackAfterCalls(Binary, LLVMFunction, GlobalSP);

  std::vector<Instruction *> SPUsers;
  for (User *U : GlobalSP->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      llvm::Function *UserFun = I->getFunction();
      revng_log(Log, "Found use in Function: " << UserFun->getName());

      if (UserFun != &LLVMFunction)
        continue;

      SPUsers.emplace_back(I);

    } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      revng_log(Log, "Found ConstantExpr use");

      if (not CE->getNumUses())
        continue;

      SmallVector<std::pair<User *, Value *>, 8> Replacements;
      for (User *CEUser : CE->users()) {
        auto *CEInstrUser = cast<Instruction>(CEUser);
        llvm::Function *UserFun = CEInstrUser->getFunction();

        if (UserFun != &LLVMFunction)
          continue;

        auto *CastInstruction = CE->getAsInstruction();
        CastInstruction->insertBefore(CEInstrUser);
        SPUsers.emplace_back(CastInstruction);
        Replacements.push_back({ CEInstrUser, CastInstruction });
      }

      for (const auto &[User, CEUseReplacement] : Replacements)
        User->replaceUsesOfWith(CE, CEUseReplacement);

    } else {
      revng_unreachable();
    }
  }

  if (SPUsers.empty())
    return;

  // Create function for initializing local stack pointer.
  Module *M = LLVMFunction.getParent();
  LLVMContext &Context = M->getContext();
  Type *SPType = GlobalSP->getValueType();
  auto *InitLocalSP = UndefinedLocalSPMarker.getOrCreate(*M, SPType).function();
  InitLocalSP->addFnAttr(Attribute::NoUnwind);
  InitLocalSP->addFnAttr(Attribute::WillReturn);
  InitLocalSP->setOnlyReadsMemory();
  FunctionTags::OpaqueCSVValue.addTo(InitLocalSP);

  // Create an alloca to represent the local value of the stack pointer.
  // This should be inserted at the beginning of the entry block.
  BasicBlock &EntryBlock = LLVMFunction.getEntryBlock();
  revng::IRBuilder Builder(Context);
  Builder.SetInsertPoint(&EntryBlock, EntryBlock.begin());
  AllocaInst *LocalSP = Builder.CreateAlloca(SPType, nullptr, "local_sp");

  // Call InitLocalSP, to initialize the value of the local stack pointer.
  Builder.setInsertPointToFirstNonAlloca(LLVMFunction);
  auto *SPVal = Builder.CreateCall(InitLocalSP);

  // Store the initial SP value in the new alloca.
  Builder.CreateStore(SPVal, LocalSP);

  // Actually perform the replacement.
  for (Instruction *I : SPUsers) {
    // Switch all the uses of the GlobalSP in I to uses of the LocalSP.
    I->replaceUsesOfWith(GlobalSP, LocalSP);
  }

  FunctionTags::StackPointerPromoted.addTo(&LLVMFunction);
}

} // namespace revng::pypeline::piperuns
