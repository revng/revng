#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Debug.h"

#include "Function.h"

namespace aua {

class ArgumentUsageAnalysis {
private:
  struct CallSite {
    llvm::Function *Callee = nullptr;
    bool IsIndirect = false;
    bool IsDeclared = false;
    bool IsVarArg = false;
    bool HasBeenAnalyzed = false;
    bool IsNoReturn = false;
    bool IsPure = false;

    template<typename O>
    void dump(O &Stream) const {
      Stream << "IsIndirect: " << IsIndirect << "\n";
      Stream << "IsDeclared: " << IsDeclared << "\n";
      Stream << "IsVarArg: " << IsVarArg << "\n";
      Stream << "HasBeenAnalyzed: " << HasBeenAnalyzed << "\n";
      Stream << "IsNoReturn: " << IsNoReturn << "\n";
      Stream << "IsPure: " << IsPure << "\n";
    }
  };

private:
  Context &TheContext;
  std::map<const llvm::Function *, Function> Results;
  llvm::Module &M;
  const llvm::DataLayout &DL;

public:
  ArgumentUsageAnalysis(Context &TheContext, llvm::Module &M) :
    TheContext(TheContext), M(M), DL(M.getDataLayout()) {}

public:
  auto begin() { return Results.begin(); }
  auto end() { return Results.end(); }

  const Function &at(const llvm::Function *F) const { return Results.at(F); }
  auto find(const llvm::Function *F) const { return Results.find(F); }
  auto begin() const { return Results.begin(); }
  auto end() const { return Results.end(); }

public:
  void run();

private:
  void analyzeFunction(llvm::Function &F);

  CallSite analyzeCallSite(const llvm::CallInst &Call);

  void taintAnalysis(Function &FunctionResults, const llvm::Function &F);

  const Value *analyzeInstruction(Function &FunctionResults,
                                  const llvm::Instruction &I);

  void registerFunctionResults(Function &FunctionResults, llvm::Instruction &I);

  void registerLoad(Function &FunctionResults, const llvm::LoadInst &Load) {
    auto Size = DL.getTypeAllocSize(Load.getType());
    registerRead(FunctionResults,
                 Load.getOperandUse(Load.getPointerOperandIndex()));
  }

  void registerRead(Function &FunctionResults, const llvm::Use &Location) {
    const Value &PointerValue = FunctionResults.get(*Location.get());
    FunctionResults.registerAccess(Location, PointerValue);
  }

  void registerStore(Function &FunctionResults, llvm::StoreInst &Store) {
    auto Size = DL.getTypeAllocSize(Store.getValueOperand()->getType());
    registerWrite(FunctionResults,
                  Store.getOperandUse(Store.getPointerOperandIndex()));

    // Register escaped value from value operand
    const Value &Value = FunctionResults.get(*Store.getValueOperand());
    FunctionResults.logEscapedValue("store " + getName(&Store), Value);
    FunctionResults.registerEscapedValue(Store, Value);
  }

  void registerWrite(Function &FunctionResults, const llvm::Use &Location) {
    const Value &PointerValue = FunctionResults.get(*Location.get());
    FunctionResults.registerAccess(Location, PointerValue);
  }

  void registerMemcpy(Function &FunctionResults, llvm::CallInst &Call) {
    registerWrite(FunctionResults, Call.getArgOperandUse(0));
    registerRead(FunctionResults, Call.getArgOperandUse(1));
  }

  void registerCall(Function &FunctionResults, llvm::CallInst &Call);

  const Value *getCallResult(const llvm::CallInst &Call);

  const Value *handleCall(Function &FunctionResults,
                          const llvm::CallInst &Call);
};

} // namespace aua
