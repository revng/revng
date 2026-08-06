#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/BasicAnalyses/RootFunction.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Support/IRHelper.h"

/// The function every indirect call is routed through
inline IRHelper<> FunctionDispatcherHelper("function_dispatcher");

namespace revng::pypeline::piperuns {

class Isolate {
private:
  /// A function to emit in the output, along with the `AlwaysInline` callees
  /// whose body has been emitted next to it so that a later pipe can inline
  /// them
  struct IsolatedFunction {
    MetaAddress Address;
    llvm::Function *Function = nullptr;
    llvm::SmallVector<llvm::Function *, 2> Inlinees;
  };

private:
  const model::Binary &Binary;
  const CFGMap &CFG;
  std::unique_ptr<llvm::Module> ClonedModule;
  LLVMFunctionContainer &Output;
  std::optional<RootFunction> RootF;
  std::optional<GeneratedCodeBasicInfo> GCBI;
  std::vector<IsolatedFunction> IsolatedFunctions;

  /// How many output modules still need the body of each isolated function
  std::map<llvm::Function *, unsigned> PendingUses;

  llvm::FunctionType *IsolatedFunctionType = nullptr;
  llvm::Function *FunctionDispatcher = nullptr;

  std::map<MetaAddress, llvm::Function *> IsolatedFunctionsMap;
  std::map<llvm::StringRef, llvm::Function *> DynamicFunctionsMap;

public:
  static constexpr llvm::StringRef Name = "isolate";
  using Arguments = TypeList<
    PipeRunArgument<const CFGMap, "CFG", "Function control flow data">,
    PipeRunArgument<LLVMRootContainer,
                    "Input",
                    "Input LLVM module to be isolated",
                    // The root container is first modified in-place to be
                    // isolated, then, as part of the destructor, the individual
                    // functions are split and put in their respective module in
                    // the LLVMFunctionContainer.
                    Access::Read>,
    PipeRunArgument<LLVMFunctionContainer,
                    "Output",
                    "Output LLVM modules with isolated functions",
                    Access::Write>>;

  Isolate(const class Model &Model,
          llvm::StringRef Config,
          llvm::StringRef DynamicConfig,
          const CFGMap &CFG,
          LLVMRootContainer &Root,
          LLVMFunctionContainer &Output);
  ~Isolate();

  void runOnFunction(const model::Function &TheFunction);

public:
  llvm::Function *getLocalFunction(const MetaAddress &Entry);
  llvm::Function *getDynamicFunction(llvm::StringRef SymbolName) const;
  llvm::Function *dispatcher() const { return FunctionDispatcher; }

private:
  /// Turn the function described by \p FM into an isolated `llvm::Function`
  ///
  /// If it has already been isolated, e.g. because it is inlined into more
  /// than one caller, the existing one is returned as is.
  llvm::Function *isolateFunction(const efa::ControlFlowGraph &FM);

  /// Drop the body of \p F if no output module needs it anymore
  void releaseBody(llvm::Function *F);

  void splitIsolatedFunctionsToOutput();

  void handleUnexpectedPCCloned(efa::OutlinedFunction &Outlined);
  void handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                        const efa::ControlFlowGraph &FM);
};

} // namespace revng::pypeline::piperuns
