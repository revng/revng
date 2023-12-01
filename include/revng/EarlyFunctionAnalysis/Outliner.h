#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/EarlyFunctionAnalysis/TemporaryOpaqueFunction.h"
#include "revng/Support/UniqueValuePtr.h"

class GeneratedCodeBasicInfo;

namespace efa {

class CallHandler;
class OutlinedFunctionsMap;

/// An outlined function helper object.
struct OutlinedFunction {
public:
  MetaAddress Address;

  /// The marker for indirect jumps
  // NOTE: do not move this field, F holds a reference to it and therefore it
  //       needs to be destroyed first.
  UniqueValuePtr<llvm::Function> IndirectBranchInfoMarker;

  /// The actual LLVM outlined function
  UniqueValuePtr<llvm::Function> Function;

  // Special blocks
  llvm::BasicBlock *AnyPCCloned = nullptr;
  llvm::BasicBlock *UnexpectedPCCloned = nullptr;

  std::vector<MetaAddress> InlinedFunctionsByIndex = { MetaAddress::invalid() };

public:
  OutlinedFunction() = default;

  OutlinedFunction(const OutlinedFunction &Other) = delete;
  OutlinedFunction &operator=(const OutlinedFunction &) = delete;

  OutlinedFunction(OutlinedFunction &&Other) { *this = std::move(Other); }

  OutlinedFunction &operator=(OutlinedFunction &&Other) {
    if (this != &Other) {
      Address = Other.Address;
      Function = std::move(Other.Function);
      IndirectBranchInfoMarker = std::move(Other.IndirectBranchInfoMarker);
      AnyPCCloned = Other.AnyPCCloned;
      UnexpectedPCCloned = Other.UnexpectedPCCloned;
      InlinedFunctionsByIndex = Other.InlinedFunctionsByIndex;

      Other.Address = MetaAddress::invalid();
      Other.Function = nullptr;
      Other.IndirectBranchInfoMarker.reset();
      Other.AnyPCCloned = nullptr;
      Other.UnexpectedPCCloned = nullptr;
    }
    return *this;
  }

public:
  llvm::Function *releaseFunction() {
    auto *ToReturn = Function.release();
    Function = nullptr;
    return ToReturn;
  }
};

/// This class, given an Oracle, can outline functions from root
class Outliner {
private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  const FunctionSummaryOracle &Oracle;

  /// UnexpectedPCMarker is used to indicate that `unexpectedpc` basic
  /// block of a function to inline need to be adjusted to jump to
  /// `unexpectedpc` of their caller.
  TemporaryOpaqueFunction UnexpectedPCMarker;

  llvm::CodeExtractorAnalysisCache CEAC;

public:
  Outliner(llvm::Module &M,
           GeneratedCodeBasicInfo &GCBI,
           const FunctionSummaryOracle &Oracle) :
    M(M),
    GCBI(GCBI),
    Oracle(Oracle),
    UnexpectedPCMarker(initializeUnexpectedPCMarker(M)),
    CEAC(*M.getFunction("root")) {}

public:
  OutlinedFunction outline(llvm::BasicBlock *BB, CallHandler *TheCallHandler);

private:
  static TemporaryOpaqueFunction initializeUnexpectedPCMarker(llvm::Module &M) {
    return { llvm::FunctionType::get(llvm::Type::getVoidTy(M.getContext()),
                                     false),
             "unexpectedpc_hook",
             &M };
  }

private:
  OutlinedFunction
  outlineFunctionInternal(CallHandler *TheCallHandler,
                          llvm::BasicBlock *BB,
                          OutlinedFunctionsMap &FunctionsToInline);

  /// \return a description of the call and boolean indicating whether the call
  ///         site is a tail call or not.
  std::pair<const FunctionSummary *, bool>
  getCallSiteInfo(CallHandler *TheCallHandler,
                  MetaAddress CallerFunction,
                  llvm::BasicBlock *CallerBlock,
                  llvm::CallInst *FunctionCall,
                  llvm::CallInst *JumpToSymbol,
                  MetaAddress Callee);

  void integrateFunctionCallee(CallHandler *TheCallHandler,
                               MetaAddress Caller,
                               llvm::BasicBlock *CallerBlock,
                               llvm::CallInst *FunctionCall,
                               llvm::CallInst *JumpToSymbol,
                               MetaAddress Callee,
                               OutlinedFunctionsMap &FunctionsToInline);

  llvm::Function *
  createFunctionToInline(CallHandler *TheCallHandler,
                         llvm::BasicBlock *BB,
                         OutlinedFunctionsMap &FunctionsToInline);

  void createAnyPCHooks(CallHandler *TheCallHandler, OutlinedFunction *F);
};

} // namespace efa
