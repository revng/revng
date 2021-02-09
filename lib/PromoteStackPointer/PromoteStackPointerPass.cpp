//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/PromoteStackPointer/PromoteStackPointerPass.h"

static Logger<> Log("promote-stack-pointer");

bool PromoteStackPointerPass::runOnFunction(llvm::Function &F) {

  // Get the global variable representing the stack pointer register.
  using GCBIPass = GeneratedCodeBasicInfoWrapperPass;
  llvm::GlobalVariable *GlobalSP = getAnalysis<GCBIPass>().getGCBI().spReg();

  std::vector<llvm::Instruction *> SPUsers;
  for (llvm::User *U : GlobalSP->users()) {
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
      llvm::Function *UserFun = I->getFunction();
      revng_log(Log, "Found use in Function: " << UserFun->getName());

      if (UserFun != &F)
        continue;

      SPUsers.emplace_back(I);

    } else if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
      revng_log(Log, "Found ConstantExpr use");

      if (not CE->getNumUses())
        continue;

      llvm::SmallVector<std::pair<llvm::User *, llvm::Value *>, 8> Replacements;
      for (llvm::User *CEUser : CE->users()) {
        auto *CEInstrUser = llvm::cast<llvm::Instruction>(CEUser);
        llvm::Function *UserFun = CEInstrUser->getFunction();

        if (UserFun != &F)
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
    return false;

  // Create function for initializing local stack pointer.
  llvm::Module *M = F.getParent();
  llvm::LLVMContext &Ctx = F.getContext();
  llvm::Type *SPType = GlobalSP->getType()->getPointerElementType();
  uint64_t PointerBitWidth = SPType->getPrimitiveSizeInBits().getFixedSize();
  llvm::Type *IntTy = llvm::Type::getIntNTy(Ctx, PointerBitWidth);
  auto InitFunction = M->getOrInsertFunction("revng_init_local_sp",
                                             SPType,
                                             IntTy);
  llvm::Function *InitLocalSP = cast<llvm::Function>(InitFunction.getCallee());

  // Create an alloca to represent the local value of the stack pointer.
  // This should be inserted at the beginning of the entry block.
  llvm::BasicBlock &EntryBlock = F.getEntryBlock();
  llvm::IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(&EntryBlock, EntryBlock.begin());
  llvm::AllocaInst *LocalSP = Builder.CreateAlloca(SPType, nullptr, "local_sp");

  // Call InitLocalSP, to initialize the value of the local stack pointer.
  // The argument to the call is always zero. This argument will be adjusted by
  // AdjustStackPointerPass to represent how much the stack pointer is adjusted.
  llvm::APInt APZero = llvm::APInt::getNullValue(IntTy->getIntegerBitWidth());
  llvm::Constant *Zero = llvm::Constant::getIntegerValue(IntTy, APZero);
  auto *InitSPVal = Builder.CreateCall(InitLocalSP, Zero);
  llvm::Type *PtrTy = llvm::PointerType::getInt8PtrTy(M->getContext());
  auto *InitSPPtr = Builder.CreateIntToPtr(InitSPVal, PtrTy);
  // Assume that the initial SP value is aligned at 4096.
  // This is not correct for the runtime semantics, but given that we plan to
  // completely decompose the stack frame we should not break anything.
  Builder.CreateAlignmentAssumption(M->getDataLayout(), InitSPPtr, 4096);
  auto *AlignedSPVal = Builder.CreatePtrToInt(InitSPPtr, InitSPVal->getType());
  // Store the initial SP value in the new alloca.
  Builder.CreateStore(AlignedSPVal, LocalSP);

  // Actually perform the replacement.
  for (llvm::Instruction *I : SPUsers) {
    // Switch all the uses of the GlobalSP in I to uses of the LocalSP.
    I->replaceUsesOfWith(GlobalSP, LocalSP);
  }

  return true;
}

char PromoteStackPointerPass::ID = 0;

using llvm::RegisterPass;
using Pass = PromoteStackPointerPass;
static RegisterPass<Pass> RegisterPromoteStackPtr("promote-stack-pointer",
                                                  "Promote Stack Pointer Pass");
