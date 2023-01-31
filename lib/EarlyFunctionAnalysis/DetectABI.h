#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"

namespace efa {

using BasicBlockQueue = UniquedQueue<const BasicBlockNode *>;

class DetectABI {
private:
  using CSVSet = std::set<llvm::GlobalVariable *>;

private:
  llvm::Module &M;
  llvm::LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  TupleTree<model::Binary> &Binary;
  FunctionSummaryOracle &Oracle;
  CFGAnalyzer &Analyzer;

  BasicBlockQueue EntrypointsQueue;

  CallGraph ApproximateCallGraph;

public:
  DetectABI(llvm::Module &M,
            GeneratedCodeBasicInfo &GCBI,
            TupleTree<model::Binary> &Binary,
            FunctionSummaryOracle &Oracle,
            CFGAnalyzer &Analyzer) :
    M(M),
    Context(M.getContext()),
    GCBI(GCBI),
    Binary(Binary),
    Oracle(Oracle),
    Analyzer(Analyzer) {}

public:
  void run();

private:
  void computeApproximateCallGraph();
  void initializeInterproceduralQueue();
  void runInterproceduralAnalysis();
  void interproceduralPropagation();
  void finalizeModel();
  void applyABIDeductions();

private:
  CSVSet computePreservedCSVs(const CSVSet &ClobberedRegisters) const;

  SortedVector<model::Register::Values>
  computePreservedRegisters(const CSVSet &ClobberedRegisters) const;
  void analyzeABI(llvm::BasicBlock *Entry);

  CSVSet findWrittenRegisters(llvm::Function *F);

  UpcastablePointer<model::Type>
  buildPrototypeForIndirectCall(const FunctionSummary &CallerSummary,
                                const efa::BasicBlock &CallerBlock);

  std::optional<abi::RegisterState::Values>
  tryGetRegisterState(model::Register::Values,
                      const ABIAnalyses::RegisterStateMap &);

  void initializeMapForDeductions(FunctionSummary &, abi::RegisterState::Map &);
};


class DetectABIPass : public llvm::ModulePass {
public:
  static char ID;

public:
  DetectABIPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};

} // namespace efa
