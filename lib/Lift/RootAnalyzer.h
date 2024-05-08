#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <unordered_set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/Model/Binary.h"
#include "revng/Support/MetaAddress.h"

#include "DropHelperCallsPass.h"

class AnalysisRegistry;
class JumpTargetManager;

class RootAnalyzer {
private:
  using MetaAddressSet = std::unordered_set<MetaAddress>;
  using Features = MetaAddress::Features;
  using GlobalToAllocaTy = llvm::DenseMap<llvm::GlobalVariable *,
                                          llvm::AllocaInst *>;

private:
  JumpTargetManager &JTM;
  llvm::Module &TheModule;
  const TupleTree<model::Binary> &Model;

public:
  RootAnalyzer(JumpTargetManager &JTM);

  void cloneOptimizeAndHarvest(llvm::Function *TheFunction);

private:
  void updateCSAA();

  llvm::Function *createTemporaryRoot(llvm::Function *TheFunction,
                                      llvm::ValueToValueMapTy &OldToNew);

  MetaAddressSet inflateValueMaterializerWhitelist();

  void promoteHelpersToIntrinsics(llvm::Function *OptimizedFunction,
                                  llvm::IRBuilder<> &Builder);

  SummaryCallsBuilder optimize(llvm::Function *OptimizedFunction,
                               const Features &CommonFeatures);

  GlobalToAllocaTy promoteCSVsToAlloca(llvm::Function *OptimizedFunction);

  void collectMaterializedValues(AnalysisRegistry &AR);

  void collectValuesStoredIntoMemory(llvm::Function *F,
                                     const Features &CommonFeatures);
};
