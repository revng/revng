#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/EarlyFunctionAnalysis/CallHandler.h"
#include "revng/EarlyFunctionAnalysis/Outliner.h"
#include "revng/EarlyFunctionAnalysis/TemporaryOpaqueFunction.h"
#include "revng/Model/Binary.h"
#include "revng/Support/OpaqueFunctionsPool.h"

class GeneratedCodeBasicInfo;
class ProgramCounterHandler;
class FunctionSummaryOracle;

namespace llvm {
class Module;
class GlobalVariable;
class Function;
} // namespace llvm

namespace efa {

class CallSummarizer : public CallHandler {
private:
  llvm::Module *M = nullptr;

  /* PreCallHook is a function returning void, taking four arguments:
   *  - MetaAddress * - caller block
   *  - MetaAddress * - callee pc
   *  - int8 * - SymbolName
   *  - bool - indicating if call is a tail call
   */
  llvm::Function *PreCallHook = nullptr;

  // PostCallHook prototype looks same as PreCallHook
  llvm::Function *PostCallHook = nullptr;

  // RetHook is returning void and taking MetaAddress *
  llvm::Function *RetHook = nullptr;

  llvm::GlobalVariable *SPCSV = nullptr;
  OpaqueFunctionsPool<llvm::StringRef> RegistersClobberedPool;

public:
  CallSummarizer(llvm::Module *M,
                 llvm::Function *PreCallHook,
                 llvm::Function *PostCallHook,
                 llvm::Function *RetHook,
                 llvm::GlobalVariable *SPCSV);

public:
  void handleCall(MetaAddress CallerBlock,
                  llvm::IRBuilder<> &Builder,
                  MetaAddress Callee,
                  const std::set<llvm::GlobalVariable *> &ClobberedRegisters,
                  const std::optional<int64_t> &MaybeFSO,
                  bool IsNoReturn,
                  bool IsTailCall,
                  llvm::Value *SymbolNamePointer) final;

  void handlePostNoReturn(llvm::IRBuilder<> &Builder) final;

  void handleIndirectJump(llvm::IRBuilder<> &Builder,
                          MetaAddress Block,
                          llvm::Value *SymbolNamePointer) final;

private:
  void clobberCSVs(llvm::IRBuilder<> &Builder,
                   const std::set<llvm::GlobalVariable *> &ClobberedRegisters);
};

/// This class, given an Oracle, analyzes a function returning its CFG, the set
/// of callee saved registers, whether it's noreturn or not and its final stack
/// offset
class CFGAnalyzer {
private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  const ProgramCounterHandler *PCH;
  const FunctionSummaryOracle &Oracle;
  const TupleTree<model::Binary> &Binary;

  llvm::SmallVector<llvm::GlobalVariable *, 16> ABICSVs;

  /// PreCallHook and PostCallHook mark the presence of an original function
  /// call, and surround a basic block containing the registers clobbered by the
  /// function called. They take the MetaAddress of the callee and the
  /// call-site.
  TemporaryOpaqueFunction PreCallHook;
  TemporaryOpaqueFunction PostCallHook;
  TemporaryOpaqueFunction RetHook;

  CallSummarizer Summarizer;
  Outliner Outliner;

  OpaqueFunctionsPool<llvm::Type *> OpaqueBranchConditionsPool;
  std::unique_ptr<llvm::raw_ostream> OutputAAWriter;
  std::unique_ptr<llvm::raw_ostream> OutputIBI;

public:
  CFGAnalyzer(llvm::Module &M,
              GeneratedCodeBasicInfo &GCBI,
              const TupleTree<model::Binary> &Binary,
              const FunctionSummaryOracle &Oracle);

public:
  llvm::Function *preCallHook() const { return PreCallHook.get(); }
  llvm::Function *postCallHook() const { return PostCallHook.get(); }
  llvm::Function *retHook() const { return RetHook.get(); }
  const auto &abiCSVs() const { return ABICSVs; }

public:
  FunctionSummary analyze(llvm::BasicBlock *Entry);

  OutlinedFunction outline(llvm::BasicBlock *Entry);

private:
  static llvm::FunctionType *createCallMarkerType(llvm::Module &M);
  static llvm::FunctionType *createRetMarkerType(llvm::Module &M);

private:
  std::optional<UpcastablePointer<efa::FunctionEdgeBase>>
  handleCall(llvm::CallInst *PreCallHookCall);

  SortedVector<efa::BasicBlock> collectDirectCFG(OutlinedFunction *F);

  void createIBIMarker(OutlinedFunction *F);

  void opaqueBranchConditions(llvm::Function *F, llvm::IRBuilder<> &);

  void materializePCValues(llvm::Function *F, llvm::IRBuilder<> &);

  void runOptimizationPipeline(llvm::Function *F);

  FunctionSummary
  milkInfo(OutlinedFunction *F, SortedVector<efa::BasicBlock> &&CFG);

  struct State {
    llvm::Value *StackPointer;
    llvm::Value *ReturnPC;
    llvm::SmallVector<llvm::Value *, 16> CSVs;
  };
  State loadState(llvm::IRBuilder<> &Builder) const;
};

} // namespace efa
