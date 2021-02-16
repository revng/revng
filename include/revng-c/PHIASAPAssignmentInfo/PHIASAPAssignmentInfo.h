#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/ADT/SmallMap.h"

struct PHIASAPAssignmentInfo : public llvm::FunctionPass {

private:
  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

public:
  static char ID;

  PHIASAPAssignmentInfo() : llvm::FunctionPass(ID), PHIInfoMap() {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  const BBPHIMap &getBBToPHIIncomingMap() const { return PHIInfoMap; }

  BBPHIMap &&extractBBToPHIIncomingMap() { return std::move(PHIInfoMap); }

private:
  BBPHIMap PHIInfoMap;
};
