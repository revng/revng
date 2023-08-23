//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
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
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/PromoteStackPointer/PromoteStackPointerPass.h"
#include "revng-c/Support/FunctionTags.h"

using namespace llvm;

static Logger<> Log("promote-stack-pointer");

static bool adjustStackAfterCalls(FunctionMetadataCache &Cache,
                                  const model::Binary &Binary,
                                  Function &F,
                                  GlobalVariable *GlobalSP) {
  bool Changed = false;

  IRBuilder<> B(F.getParent()->getContext());

  Type *SPType = GlobalSP->getValueType();

  MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
  auto &ModelFunction = Binary.Functions().at(Entry);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isCallToIsolatedFunction(&I)) {
        // TODO: handle CABIFunctionType
        auto *Proto = Cache
                        .getCallSitePrototype(Binary,
                                              cast<CallInst>(&I),
                                              &ModelFunction)
                        .get();

        using Layout = abi::FunctionType::Layout;
        auto TheLayout = Layout::make(*Proto);
        auto *FSO = ConstantInt::get(SPType, TheLayout.FinalStackOffset);

        // We found a function call
        Changed = true;

        B.SetInsertPoint(I.getNextNode());
        B.CreateStore(B.CreateAdd(createLoad(B, GlobalSP), FSO), GlobalSP);
      }
    }
  }

  return Changed;
}

bool PromoteStackPointerPass::runOnFunction(Function &F) {
  bool Changed = false;

  {
    // A couple of preliminary assertions
    using namespace FunctionTags;
    revng_assert(TagsSet::from(&F).contains(Isolated));
    revng_assert(not F.isDeclaration());
  }

  // Get the global variable representing the stack pointer register.
  using GCBIPass = GeneratedCodeBasicInfoWrapperPass;
  GlobalVariable *GlobalSP = getAnalysis<GCBIPass>().getGCBI().spReg();

  if (not GlobalSP) {
    revng_log(Log, "WARNING: cannot find global variable for stack pointer");
    return Changed;
  }

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();

  Changed = adjustStackAfterCalls(getAnalysis<FunctionMetadataCachePass>()
                                    .get(),
                                  Binary,
                                  F,
                                  GlobalSP)
            or Changed;

  std::vector<Instruction *> SPUsers;
  for (User *U : GlobalSP->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      Function *UserFun = I->getFunction();
      revng_log(Log, "Found use in Function: " << UserFun->getName());

      if (UserFun != &F)
        continue;

      SPUsers.emplace_back(I);

    } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      revng_log(Log, "Found ConstantExpr use");

      if (not CE->getNumUses())
        continue;

      SmallVector<std::pair<User *, Value *>, 8> Replacements;
      for (User *CEUser : CE->users()) {
        auto *CEInstrUser = cast<Instruction>(CEUser);
        Function *UserFun = CEInstrUser->getFunction();

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
    return Changed;

  // Create function for initializing local stack pointer.
  Module *M = F.getParent();
  LLVMContext &Ctx = F.getContext();
  Type *SPType = GlobalSP->getValueType();
  auto InitFunction = M->getOrInsertFunction("_init_local_sp", SPType);
  Function *InitLocalSP = cast<Function>(InitFunction.getCallee());
  InitLocalSP->addFnAttr(Attribute::NoUnwind);
  InitLocalSP->addFnAttr(Attribute::WillReturn);
  InitLocalSP->setOnlyAccessesInaccessibleMemory();
  FunctionTags::OpaqueCSVValue.addTo(InitLocalSP);

  // Create an alloca to represent the local value of the stack pointer.
  // This should be inserted at the beginning of the entry block.
  BasicBlock &EntryBlock = F.getEntryBlock();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(&EntryBlock, EntryBlock.begin());
  AllocaInst *LocalSP = Builder.CreateAlloca(SPType, nullptr, "local_sp");

  // Call InitLocalSP, to initialize the value of the local stack pointer.
  setInsertPointToFirstNonAlloca(Builder, F);
  auto *SPVal = Builder.CreateCall(InitLocalSP);

  // Store the initial SP value in the new alloca.
  Builder.CreateStore(SPVal, LocalSP);

  // Actually perform the replacement.
  for (Instruction *I : SPUsers) {
    // Switch all the uses of the GlobalSP in I to uses of the LocalSP.
    I->replaceUsesOfWith(GlobalSP, LocalSP);
  }

  return true;
}

void PromoteStackPointerPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
  AU.setPreservesCFG();
}

char PromoteStackPointerPass::ID = 0;

static constexpr const char *Flag = "promote-stack-pointer";

using Reg = RegisterPass<PromoteStackPointerPass>;
static Reg Register(Flag, "Promote Stack Pointer Pass");

struct PromoteStackPointerPipe {
  static constexpr auto Name = Flag;

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;
    return { ContractGroup::transformOnlyArgument(LiftingArtifactsRemoved,
                                                  StackPointerPromoted,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(legacy::PassManager &Manager) {
    Manager.add(new PromoteStackPointerPass());
  }
};

static pipeline::RegisterLLVMPass<PromoteStackPointerPipe> Y;
