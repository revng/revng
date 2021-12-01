/// \file CSVAliasAnalysis.cpp
/// \brief Decorate memory accesses with information about CSV aliasing.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"

#include "revng/BasicAnalyses/CSVAliasAnalysis.h"
#include "revng/Support/Assert.h"

using namespace llvm;
class CSVAliasAnalysisPassImpl;

using CSVAAP = CSVAliasAnalysisPass;
using CSVAAPI = CSVAliasAnalysisPassImpl;

class CSVAliasAnalysisPassImpl {
  struct CSVAliasInfo {
    MDNode *AliasScope;
    MDNode *AliasSet;
    MDNode *NoAliasSet;
  };
  std::map<const GlobalVariable *, CSVAliasInfo> CSVAliasInfoMap;
  std::vector<Metadata *> AllCSVScopes;
  GeneratedCodeBasicInfo *GCBI = nullptr;

public:
  void run(Module &, ModuleAnalysisManager &);

private:
  void initializeAliasInfo(Module &M);
  void decorateMemoryAccesses(Instruction &I);
};

PreservedAnalyses CSVAAP::run(Module &M, ModuleAnalysisManager &MAM) {
  CSVAliasAnalysisPassImpl CSVAAP;
  CSVAAP.run(M, MAM);
  return PreservedAnalyses::none();
}

void CSVAAPI::run(Module &M, ModuleAnalysisManager &MAM) {
  // Get the result of the GCBI analysis
  GCBI = &(MAM.getResult<GeneratedCodeBasicInfoAnalysis>(M));
  revng_assert(GCBI != nullptr);

  // Initialize the alias information for the CSVs.
  initializeAliasInfo(M);

  // Decorate the IR with the alias information for the CSVs.
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        decorateMemoryAccesses(I);
}

void CSVAAPI::initializeAliasInfo(Module &M) {
  LLVMContext &Context = M.getContext();
  QuickMetadata QMD(Context);
  MDBuilder MDB(Context);

  MDNode *AliasDomain = MDB.createAliasScopeDomain("CSVAliasDomain");
  std::vector<GlobalVariable *> CSVs = GCBI->csvs();

  const auto *PCH = GCBI->programCounterHandler();
  for (GlobalVariable *PCCSV : PCH->pcCSVs())
    CSVs.emplace_back(PCCSV);

  // Build alias scopes
  for (const GlobalVariable *CSV : CSVs) {
    CSVAliasInfo &AliasInfo = CSVAliasInfoMap[CSV];

    MDNode *CSVScope = MDB.createAliasScope(CSV->getName(), AliasDomain);
    AliasInfo.AliasScope = CSVScope;
    AllCSVScopes.push_back(CSVScope);
    MDNode *CSVAliasSet = MDNode::get(Context,
                                      ArrayRef<Metadata *>({ CSVScope }));
    AliasInfo.AliasSet = CSVAliasSet;
  }

  // Build noalias sets
  for (const GlobalVariable *CSV : CSVs) {
    CSVAliasInfo &AliasInfo = CSVAliasInfoMap[CSV];
    std::vector<Metadata *> OtherCSVScopes;
    for (const auto &Q : CSVAliasInfoMap)
      if (Q.first != CSV)
        OtherCSVScopes.push_back(Q.second.AliasScope);

    MDNode *CSVNoAliasSet = MDNode::get(Context, OtherCSVScopes);
    AliasInfo.NoAliasSet = CSVNoAliasSet;
  }
}

void CSVAAPI::decorateMemoryAccesses(Instruction &I) {
  Value *Pointer = nullptr;

  if (auto *L = dyn_cast<LoadInst>(&I))
    Pointer = L->getPointerOperand();
  else if (auto *S = dyn_cast<StoreInst>(&I))
    Pointer = S->getPointerOperand();
  else
    return;

  // Check if the pointer is a CSV
  if (auto *GV = dyn_cast<GlobalVariable>(Pointer)) {
    auto It = CSVAliasInfoMap.find(GV);
    if (It != CSVAliasInfoMap.end()) {
      // Set alias.scope and noalias metadata
      auto *CurrentMDAliasScope = I.getMetadata(LLVMContext::MD_alias_scope);
      auto *CurrentMDNoAliasScope = I.getMetadata(LLVMContext::MD_noalias);
      I.setMetadata(LLVMContext::MD_alias_scope, It->second.AliasSet);
      I.setMetadata(LLVMContext::MD_noalias, It->second.NoAliasSet);
      return;
    }
  }

  // It's not a CSV memory access, set noalias info
  MDNode *MemoryAliasSet = MDNode::get(I.getContext(), AllCSVScopes);
  I.setMetadata(LLVMContext::MD_noalias, MemoryAliasSet);
}
