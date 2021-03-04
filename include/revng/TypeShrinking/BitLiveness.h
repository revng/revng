#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/IR/PassManager.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/TypeShrinking/DataFlowGraph.h"
#include "revng/TypeShrinking/MFP.h"

namespace llvm {
class Instruction;
} // end namespace llvm

namespace TypeShrinking {

extern const uint32_t Top;

bool isDataFlowSink(const llvm::Instruction *Ins);

/// This class is an instance of monotone framework
/// the elements represent the index from which all bits are not alive
/// so for an element E, all bits with index < E are alive
struct BitLivenessAnalysis {
  using GraphType = GenericGraph<DataFlowNode> *;
  using LatticeElement = uint32_t;
  using Label = DataFlowNode *;
  using MFPResult = MFPResult<BitLivenessAnalysis::LatticeElement>;
  static uint32_t combineValues(const uint32_t &Lh, const uint32_t &Rh);
  static uint32_t applyTransferFunction(DataFlowNode *L, const uint32_t E);
  static bool isLessOrEqual(const uint32_t &Lh, const uint32_t &Rh);
};

inline uint32_t
BitLivenessAnalysis::combineValues(const uint32_t &Lh, const uint32_t &Rh) {
  return std::max(Lh, Rh);
}

inline bool
BitLivenessAnalysis::isLessOrEqual(const uint32_t &Lh, const uint32_t &Rh) {
  return Lh <= Rh;
}

using Label = BitLivenessAnalysis::Label;
using AnalysisResult = std::map<Label, BitLivenessAnalysis::MFPResult>;

class BitLivenessWrapperPass : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  BitLivenessWrapperPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  AnalysisResult &getResult() { return Result; }

private:
  AnalysisResult Result;
};

class BitLivenessPass : public llvm::AnalysisInfoMixin<BitLivenessPass> {
  friend llvm::AnalysisInfoMixin<BitLivenessPass>;

private:
  static llvm::AnalysisKey Key;

public:
  AnalysisResult run(llvm::Function &F, llvm::FunctionAnalysisManager &);
};

} // namespace TypeShrinking
